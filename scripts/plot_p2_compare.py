# scripts/plot_p2_compare.py
# ----------------------------------------------------------
# P2: Power minimization under SINR targets Γ
#
# This script:
#   - Uses SystemConfig for all system parameters
#   - Generates channels with 3GPP UMa + small-scale fading
#   - For each SINR target Γ (in dB), runs Monte Carlo:
#         * P2 convex optimal (p2_min_power)
#         * ZF-based heuristic with fixed directions + power bisection
#   - Averages total power vs Γ
#   - Saves:
#         results/results_p2_compare.csv
#         results/fig_p2_compare.png
#
# Run from the project root as:
#   python -m scripts.plot_p2_compare --N_mc 2000 --gammas "-5,0,5,10,15,20"  / 
# ------------------------- python -m scripts.plot_p2_compare --N_mc 2000 --gammas "-5 0 5 10 15 20"

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import SystemConfig
from scripts.common import (
    make_rng,
    draw_iid_small_scale,
    sinr_vector,
    zf_equal_power,
    p2_min_power,
)
from sim.sionna_channel import generate_h_rb_true
from scripts.evaluate_p2 import evaluate_p2


# ----------------------------------------------------------
# Helper: ZF directions + power scaling via bisection
# ----------------------------------------------------------
def zf_min_power_fixed_directions(H, sigma2, Pmax, gamma_target_lin):
    """
    Heuristic P2:
      1) Compute ZF directions (unit-norm columns)
      2) Scale all powers by a factor t (shared by all users)
         via bisection so that min_k SINR_k >= gamma_target_lin
         if possible under total power <= Pmax.

    Returns:
      (total_power, feasible_flag)
    """
    U, M = H.shape
    if U == 0:
        return 0.0, False

    # 1) Get ZF equal-power precoder (already in scripts.common)
    W_eq = zf_equal_power(H, Pmax)  # [M,U], total power = Pmax

    # 2) Extract pure directions V (each column norm = 1)
    p_each = Pmax / max(U, 1)
    if p_each <= 0.0:
        return 0.0, False

    V = W_eq / np.sqrt(p_each)  # now ||V[:,k]||^2 = 1
    base_power_per_user = 1.0   # each column has power t after scaling

    # Total power after scaling by t: P_tot = U * t  <= Pmax
    t_hi = Pmax / (U * base_power_per_user + 1e-12)
    t_lo = 0.0

    def min_sinr_for_t(t):
        W = np.sqrt(t) * V
        sinr = sinr_vector(H, W, sigma2)
        return float(np.min(sinr))

    # Quick feasibility check at maximum t
    if min_sinr_for_t(t_hi) < gamma_target_lin:
        W_hi = np.sqrt(t_hi) * V
        return U * t_hi, False, W_hi  # infeasible under these directions

    # Bisection on t in [t_lo, t_hi]
    for _ in range(50):
        t_mid = 0.5 * (t_lo + t_hi)
        if t_mid <= 0.0:
            t_lo = t_mid
            continue
        sinr_min = min_sinr_for_t(t_mid)
        if sinr_min >= gamma_target_lin:
            t_hi = t_mid
        else:
            t_lo = t_mid
        if abs(t_hi - t_lo) / max(t_hi, 1e-12) < 1e-3:
            break

    t_star = t_hi
    total_power = U * t_star
    W_star = np.sqrt(t_star) * V
    return total_power, True, W_star


def evaluate_single_rb_with_p2_core(cfg: SystemConfig, H_sub, W_sub, gamma_lin):
    """
    Evaluate objective and feasibility through evaluate_p2 core for a single RB.
    H_sub: [U,M], W_sub: [M,U]
    """
    U, M = H_sub.shape
    K_eval = U
    h_true = tf.convert_to_tensor(H_sub.reshape(1, K_eval, 1, M), dtype=tf.complex64)
    sel = tf.convert_to_tensor(np.arange(U, dtype=np.int32).reshape(1, 1, U))
    W = tf.convert_to_tensor(W_sub.reshape(1, 1, M, U), dtype=tf.complex64)
    R_req = float(cfg.B_RB) * np.log2(1.0 + float(gamma_lin))
    R_th = tf.fill([K_eval], tf.constant(R_req, tf.float32))

    out = evaluate_p2(
        h_true=h_true,
        sel=sel,
        W=W,
        R_th=R_th,
        sigma2=tf.constant(float(cfg.sigma2), tf.float32),
        B_RB=tf.constant(float(cfg.B_RB), tf.float32),
        P_tot=tf.constant(float(cfg.P_tot), tf.float32),
        U_max=min(cfg.U_max, cfg.M),
        rate_per_hz=False,
        require_all_users=True,
        r_max=1,
        apply_channel_conjugate=False,
    )
    p_obj = float(out["objective_power"][0].numpy())
    feasible = bool(out["feasible"][0].numpy())
    return p_obj, feasible


# ----------------------------------------------------------
# Main Monte Carlo loop
# ----------------------------------------------------------
def run_p2_compare(
    cfg: SystemConfig,
    N_mc: int = 1000,
    gamma_dB_list=None,
    seed: int | None = None,
):
    if gamma_dB_list is None:
        gamma_dB_list = [-5, 0, 5, 10, 15, 20]

    rng = make_rng(cfg.seed if seed is None else seed)

    K = cfg.K
    M = cfg.M
    sigma2 = cfg.sigma2
    Pmax = cfg.P_tot

    # For P2 we use a fixed number of users <= U_max and <= M.
    U = min(cfg.U_max, K, M)

    # Pre-generate 3GPP UMa channels via Sionna (RB-level) if requested.
    # We select one RB index (default 0) to match the single-RB evaluation here.
    if seed is not None:
        tf.random.set_seed(seed)
    h_rb_true = generate_h_rb_true(cfg, batch_size=N_mc)  # [B,K,N_RB,Nt]
    h_rb_true_np = h_rb_true.numpy()
    rb_idx = int(getattr(cfg, "rb_index", 0))
    if rb_idx < 0 or rb_idx >= int(cfg.N_RB):
        rb_idx = 0
    HkM_eff_samples = h_rb_true_np[:, :, rb_idx, :]  # [B,K,M]

    results = {
        "gamma_dB": [],
        "P2_opt_avg": [],
        "P2_zf_avg": [],
        "P2_opt_infeas": [],
        "P2_zf_infeas": [],
    }

    for gamma_dB in gamma_dB_list:
        gamma_lin = 10.0 ** (gamma_dB / 10.0)
        print(f"Γ = {gamma_dB:.1f} dB  (γ = {gamma_lin:.3e})")

        powers_opt_all = []
        powers_zf_all = []
        infeas_opt = 0
        infeas_zf = 0

        for s in range(N_mc):
            # ----- Channel generation -----
            # 1) Small-scale
            Hs = draw_iid_small_scale(rng, K, M)  # [K,M]

            # 2) Large-scale + small-scale (3GPP UMa via Sionna)
            H_eff = HkM_eff_samples[s]

            # ----- Select a subset of U users (same for both methods) -----
            if U == K:
                idx = np.arange(K, dtype=np.int32)
            else:
                idx = rng.choice(K, size=U, replace=False)
            H_sub = H_eff[idx, :]  # [U,M]

            # ----- P2 optimal via convex-like solver (p2_min_power) -----
            gamma_vec = np.full(U, gamma_lin, dtype=np.float64)
            total_power_opt, p_opt, W_opt = p2_min_power(H_sub, gamma_vec, sigma2)
            if (not np.isfinite(total_power_opt)) or (W_opt is None):
                infeas_opt += 1
                powers_opt_all.append(Pmax)
            else:
                # Keep original solver feasibility logic for reproducibility.
                # evaluate_p2 core is used to compute/validate objective value.
                p_eval, _ = evaluate_single_rb_with_p2_core(cfg, H_sub, W_opt, gamma_lin)
                powers_opt_all.append(p_eval)

            # ----- ZF-based heuristic -----
            total_power_zf, feas_zf, W_zf = zf_min_power_fixed_directions(
                H_sub, sigma2, Pmax, gamma_lin
            )
            p_eval_zf, _ = evaluate_single_rb_with_p2_core(cfg, H_sub, W_zf, gamma_lin)
            if not feas_zf:
                infeas_zf += 1
                powers_zf_all.append(Pmax)
            else:
                powers_zf_all.append(p_eval_zf)

        def safe_mean(arr):
            if len(arr) == 0:
                return np.nan
            return float(np.mean(arr))

        results["gamma_dB"].append(gamma_dB)
        results["P2_opt_avg"].append(safe_mean(powers_opt_all))
        results["P2_zf_avg"].append(safe_mean(powers_zf_all))
        results["P2_opt_infeas"].append(infeas_opt)
        results["P2_zf_infeas"].append(infeas_zf)

        print(
            f"  -> P2_opt: meanP={results['P2_opt_avg'][-1]:.6f} W, "
            f"infeasible={infeas_opt}/{N_mc}"
        )
        print(
            f"  -> ZF-heur: meanP={results['P2_zf_avg'][-1]:.6f} W, "
            f"infeasible={infeas_zf}/{N_mc}"
        )

    return results


# ----------------------------------------------------------
# CSV + Figure saving
# ----------------------------------------------------------
def save_results(results, out_dir="results", cfg: SystemConfig | None = None):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results_p2_compare.csv")

    # ---- CSV ----
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gamma_dB",
                "P2_opt_avg_power",
                "P2_zf_avg_power",
                "P2_opt_infeasible_count",
                "P2_zf_infeasible_count",
            ]
        )
        for i in range(len(results["gamma_dB"])):
            writer.writerow(
                [
                    results["gamma_dB"][i],
                    results["P2_opt_avg"][i],
                    results["P2_zf_avg"][i],
                    results["P2_opt_infeas"][i],
                    results["P2_zf_infeas"][i],
                ]
            )

    print("Saved CSV:", csv_path)

    # ---- Figure ----
    fig_path = os.path.join(out_dir, "fig_p2_compare.png")

    gamma = np.array(results["gamma_dB"])
    Popt = np.array(results["P2_opt_avg"])
    Pzf = np.array(results["P2_zf_avg"])

    plt.figure()
    plt.plot(gamma, Popt, marker="o", label="P2 optimal (capped mean)")
    plt.plot(gamma, Pzf, marker="s", label="ZF heuristic (capped mean)")
    plt.xlabel("Target SINR Γ (dB)")
    plt.ylabel("Average total transmit power (W)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()

    title = "P2: Power minimization vs SINR target"
    if cfg is not None:
        U = min(cfg.U_max, cfg.K, cfg.M)
        title += f"\nM={cfg.M}, K={cfg.K}, U={U}, P_tot={cfg.P_tot:.1f} W"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("Saved figure:", fig_path)


# ----------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------
def parse_gamma_list(s: str):
    """
    Parse a flexible list of gamma_dB values:
      examples: "-5,0,5", "-5 0 5", "-5;0;5"
    """
    parts = re.split(r"[,\s;]+", s.strip())
    return [float(x) for x in parts if x]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_mc", type=int, default=1000,
                        help="Number of Monte Carlo channel realizations")
    parser.add_argument(
        "--gammas",
        type=str,
        default="-5,0,5,10,15,20",
        help='List of target SINR in dB, e.g. "-5,0,5,10" or "-5 0 5 10"',
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed (overrides cfg.seed)")

    parser.add_argument("--outdir", type=str, default="results",
                        help="Output directory for CSV and figure")
    args = parser.parse_args()

    cfg = SystemConfig()

    gamma_list = parse_gamma_list(args.gammas)
    print("Using gamma_dB list:", gamma_list)
    print("SystemConfig:", cfg)

    results = run_p2_compare(cfg, N_mc=args.N_mc,
                             gamma_dB_list=gamma_list,
                             seed=args.seed)
    save_results(results, out_dir=args.outdir, cfg=cfg)


if __name__ == "__main__":
    main()
