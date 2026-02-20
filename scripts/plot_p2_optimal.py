# scripts/plot_p2_optimal.py
# ----------------------------------------------------------
# P2: Power minimization under SINR targets Γ (OPTIMAL ONLY)
#
# This script:
#   - Uses SystemConfig for all system parameters
#   - Generates channels with 3GPP UMa + small-scale fading
#   - For each SINR target Γ (in dB), runs Monte Carlo:
#         * P2 convex optimal (p2_min_power)
#   - Averages total power vs Γ (over feasible realizations)
#   - Saves:
#         results/results_p2_optimal.csv
#         results/fig_p2_optimal.png
#
# Run from the project root as:
#   python -m scripts.plot_p2_optimal --N_mc 2000 --gammas "-5,0,5,10,15,20"
#   python -m scripts.plot_p2_optimal --N_mc 2000 --gammas "-5 0 5 10 15 20"
# ----------------------------------------------------------

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

from config import SystemConfig
from scripts.common import (
    make_rng,
    draw_iid_small_scale,
    effective_channel,
    p2_min_power,
)
from sim.channels_3gpp import gen_large_scale_3gpp_uma


# ----------------------------------------------------------
# Main Monte Carlo loop (Optimal only)
# ----------------------------------------------------------
def run_p2_optimal(
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

    # For P2 we use a fixed number of users <= U_max and <= M.
    U = min(cfg.U_max, K, M)

    results = {
        "gamma_dB": [],
        "P2_opt_avg": [],
        "P2_opt_infeas": [],
        "P2_opt_feas_count": [],
    }

    for gamma_dB in gamma_dB_list:
        gamma_lin = 10.0 ** (gamma_dB / 10.0)
        print(f"Γ = {gamma_dB:.1f} dB  (γ = {gamma_lin:.3e})")

        powers_opt = []
        infeas_opt = 0

        for _ in range(N_mc):
            # ----- Channel generation -----
            # 1) Small-scale
            Hs = draw_iid_small_scale(rng, K, M)  # [K,M]

            # 2) Large-scale 3GPP UMa
            beta, is_los, d2d_m = gen_large_scale_3gpp_uma(
                K=K,
                fc_GHz=cfg.fc_GHz,
                h_bs=cfg.h_bs,
                h_ut=cfg.h_ut,
                los_mode=cfg.los_mode,
                cell_radius_m=cfg.cell_radius_m,
                rng=rng,
            )

            H_eff = effective_channel(Hs, beta)  # [K,M]

            # ----- Select a subset of U users -----
            if U == K:
                idx = np.arange(K, dtype=np.int32)
            else:
                idx = rng.choice(K, size=U, replace=False)
            H_sub = H_eff[idx, :]  # [U,M]

            # ----- P2 optimal -----
            gamma_vec = np.full(U, gamma_lin, dtype=np.float64)
            total_power_opt, p_opt, W_opt = p2_min_power(H_sub, gamma_vec, sigma2)

            if (not np.isfinite(total_power_opt)) or (W_opt is None):
                infeas_opt += 1
            else:
                powers_opt.append(float(total_power_opt))

        feas_count = len(powers_opt)

        def safe_mean(arr):
            return float(np.mean(arr)) if len(arr) > 0 else np.nan

        results["gamma_dB"].append(float(gamma_dB))
        results["P2_opt_avg"].append(safe_mean(powers_opt))
        results["P2_opt_infeas"].append(int(infeas_opt))
        results["P2_opt_feas_count"].append(int(feas_count))

        print(
            f"  -> P2_opt: meanP={results['P2_opt_avg'][-1]:.6f} W, "
            f"feasible={feas_count}/{N_mc}, infeasible={infeas_opt}/{N_mc}"
        )

    return results


# ----------------------------------------------------------
# CSV + Figure saving
# ----------------------------------------------------------
def save_results(results, out_dir="results", cfg: SystemConfig | None = None):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results_p2_optimal.csv")

    # ---- CSV ----
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gamma_dB",
                "P2_opt_avg_power",
                "P2_opt_feasible_count",
                "P2_opt_infeasible_count",
            ]
        )
        for i in range(len(results["gamma_dB"])):
            writer.writerow(
                [
                    results["gamma_dB"][i],
                    results["P2_opt_avg"][i],
                    results["P2_opt_feas_count"][i],
                    results["P2_opt_infeas"][i],
                ]
            )

    print("Saved CSV:", csv_path)

    # ---- Figure ----
    fig_path = os.path.join(out_dir, "fig_p2_optimal.png")

    gamma = np.array(results["gamma_dB"], dtype=float)
    Popt = np.array(results["P2_opt_avg"], dtype=float)

    plt.figure()
    plt.plot(gamma, Popt, marker="o", label="P2 optimal (convex)")
    plt.xlabel("Target SINR Γ (dB)")
    plt.ylabel("Average total transmit power (W)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()

    title = "P2: Optimal power minimization vs SINR target"
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

    results = run_p2_optimal(
        cfg,
        N_mc=args.N_mc,
        gamma_dB_list=gamma_list,
        seed=args.seed,
    )
    save_results(results, out_dir=args.outdir, cfg=cfg)


if __name__ == "__main__":
    main()
