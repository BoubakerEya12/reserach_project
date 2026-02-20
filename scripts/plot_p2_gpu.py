# scripts/plot_p2_gpu.py
# ----------------------------------------------------------
# P2: Power minimization under SINR targets Γ (GPU-oriented)
# Same methods as plot_p2_compare.py:
#   - P2 optimal (GPU exact-subset over fixed beam families)
#   - ZF-based heuristic (GPU, top-U + fixed directions)
#
# Notes:
# - Channels are generated in batches with Sionna (TensorFlow).
# - Evaluation uses scripts.evaluate_p2 (TensorFlow, vector/tensor based).
# ----------------------------------------------------------

import os
import re
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import SystemConfig
from sim.sionna_channel import generate_h_rb_true, setup_gpu_memory_growth
from scripts.evaluate_p2 import evaluate_p2


def parse_gamma_list(s: str):
    parts = re.split(r"[,\s;]+", s.strip())
    return [float(x) for x in parts if x]


def _build_subset_cache_tf(K: int, U_max: int, fixed_u: int | None = None):
    cache = {}
    if fixed_u is not None:
        u_list = [int(fixed_u)]
    else:
        u_list = list(range(1, min(U_max, K) + 1))
    for u in u_list:
        comb = np.array(list(__import__("itertools").combinations(range(K), u)), dtype=np.int32)
        cache[u] = tf.convert_to_tensor(comb, dtype=tf.int32)
    return cache


def _directions_from_hsel(H_sel: tf.Tensor, mode: str, sigma2: float, Pmax: float):
    # H_sel: [C,U,M] -> V:[C,M,U]
    c = tf.shape(H_sel)[0]
    u = tf.shape(H_sel)[1]
    H = tf.transpose(H_sel, [0, 2, 1])                     # [C,M,U]
    reg = tf.cast(1e-6, H.dtype)
    eye = tf.eye(u, batch_shape=[c], dtype=H.dtype)

    if mode == "mrt":
        V = tf.math.conj(H)
    else:
        Gram = tf.matmul(tf.linalg.adjoint(H), H)          # [C,U,U]
        if mode == "zf":
            A = Gram + reg * eye
        else:  # rzf
            alpha = tf.cast(u, tf.float32) * tf.cast(sigma2, tf.float32) / tf.maximum(tf.cast(Pmax, tf.float32), 1e-12)
            A = Gram + tf.cast(alpha, Gram.dtype) * eye + reg * eye
        V = tf.matmul(H, tf.linalg.inv(A))

    V = V / (tf.norm(V, axis=1, keepdims=True) + 1e-12)
    return tf.cast(V, tf.complex64)


def _min_power_for_fixed_dirs(H_sel: tf.Tensor, V: tf.Tensor, gamma_lin: float, sigma2: float, Pmax: float):
    # H_sel:[C,U,M], V:[C,M,U]
    HV = tf.matmul(H_sel, V)                                    # [C,U,U]
    G = tf.abs(HV) ** 2
    gkk = tf.maximum(tf.linalg.diag_part(G), tf.constant(1e-12, G.dtype))
    gamma = tf.cast(gamma_lin, G.dtype)

    F = gamma * (G / gkk[:, :, None])
    F = F - tf.linalg.diag(tf.linalg.diag_part(F))
    uvec = gamma * tf.cast(sigma2, G.dtype) / gkk

    c = tf.shape(H_sel)[0]
    u = tf.shape(H_sel)[1]
    I = tf.eye(u, batch_shape=[c], dtype=G.dtype)
    p_raw = tf.math.real(tf.linalg.solve(I - F, uvec[..., None])[..., 0])   # [C,U]
    finite = tf.reduce_all(tf.math.is_finite(p_raw), axis=1)
    nonneg = tf.reduce_all(p_raw >= -1e-5, axis=1)
    p = tf.maximum(p_raw, 0.0)

    # Validate achieved SINR with the computed powers.
    gains_p = G * p[:, None, :]                                    # [C,U,U]
    sig = tf.linalg.diag_part(gains_p)
    interf = tf.reduce_sum(gains_p, axis=-1) - sig
    sinr = sig / (interf + tf.cast(sigma2, sig.dtype))
    sinr_ok = tf.reduce_all(sinr >= tf.cast(gamma_lin * (1.0 - 5e-3), sinr.dtype), axis=1)

    p_tot = tf.reduce_sum(p, axis=1)
    feas = finite & nonneg & sinr_ok & (p_tot <= tf.cast(Pmax, p_tot.dtype) * 1.0001)
    W = V * tf.cast(tf.sqrt(p[:, None, :]), V.dtype)
    return p_tot, feas, W


def _best_power_exact_subset_tf(HkM: tf.Tensor, gamma_lin: float, sigma2: float, Pmax: float, subset_cache):
    """
    Exact subset search on GPU for fixed beam families (ZF/RZF/MRT).
    """
    best_p = tf.constant(float(Pmax), tf.float32)
    best_feas = tf.constant(False)

    for u, sel_all in subset_cache.items():
        H_sel = tf.gather(HkM, sel_all, axis=0)                 # [C,U,M]
        for mode in ("zf", "rzf", "mrt"):
            V = _directions_from_hsel(H_sel, mode=mode, sigma2=sigma2, Pmax=Pmax)
            p_tot, feas, _ = _min_power_for_fixed_dirs(H_sel, V, gamma_lin, sigma2, Pmax)
            masked = tf.where(feas, tf.cast(p_tot, tf.float32), tf.fill(tf.shape(p_tot), tf.constant(float(Pmax), tf.float32)))
            cand = tf.reduce_min(masked)
            best_p = tf.minimum(best_p, cand)
            best_feas = best_feas | tf.reduce_any(feas)

    return float(best_p.numpy()), bool(best_feas.numpy())


def _zf_heur_power_tf(HkM: tf.Tensor, gamma_lin: float, sigma2: float, Pmax: float, U: int):
    norms = tf.math.real(tf.reduce_sum(HkM * tf.math.conj(HkM), axis=1))
    idx = tf.math.top_k(norms, k=U, sorted=True).indices
    H_sel = tf.gather(HkM, idx, axis=0)[None, :, :]            # [1,U,M]
    V = _directions_from_hsel(H_sel, mode="zf", sigma2=sigma2, Pmax=Pmax)
    p_tot, feas, W = _min_power_for_fixed_dirs(H_sel, V, gamma_lin, sigma2, Pmax)
    return float(p_tot[0].numpy()), bool(feas[0].numpy()), idx, W[0]


def _zf_heur_power_given_subset_tf(H_sub: tf.Tensor, gamma_lin: float, sigma2: float, Pmax: float):
    # H_sub: [U,M]
    H_sel = H_sub[None, :, :]                                   # [1,U,M]
    V = _directions_from_hsel(H_sel, mode="zf", sigma2=sigma2, Pmax=Pmax)
    p_tot, feas, W = _min_power_for_fixed_dirs(H_sel, V, gamma_lin, sigma2, Pmax)
    return float(p_tot[0].numpy()), bool(feas[0].numpy()), W[0]


def evaluate_single_rb_with_p2_core(cfg: SystemConfig, H_sub, W_sub, gamma_lin):
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
    return float(out["objective_power"][0].numpy()), bool(out["feasible"][0].numpy())


def run_p2_gpu(cfg: SystemConfig,
               N_mc: int = 1000,
               gamma_dB_list=None,
               seed: int | None = None,
               chunk_size: int = 512):
    if gamma_dB_list is None:
        gamma_dB_list = [-5, 0, 5, 10, 15, 20]

    if seed is not None:
        tf.random.set_seed(seed)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    K = cfg.K
    M = cfg.M
    sigma2 = cfg.sigma2
    Pmax = cfg.P_tot
    U = min(cfg.U_max, K, M)
    # Keep same problem setting as legacy compare: fixed multiplexing order U.
    subset_cache = _build_subset_cache_tf(K, U, fixed_u=U)

    rb_idx = int(getattr(cfg, "rb_index", 0))
    if rb_idx < 0 or rb_idx >= int(cfg.N_RB):
        rb_idx = 0

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

        done = 0
        chunk_id = 0
        while done < N_mc:
            bs = min(chunk_size, N_mc - done)
            if seed is not None:
                tf.random.set_seed(seed + 1000 * chunk_id)
            h_rb_true = generate_h_rb_true(cfg, batch_size=bs)
            HkM_eff_samples = h_rb_true[:, :, rb_idx, :]  # [bs,K,M]

            for s in range(bs):
                H_eff = HkM_eff_samples[s]

                total_power_opt, feas_opt = _best_power_exact_subset_tf(
                    HkM=H_eff,
                    gamma_lin=gamma_lin,
                    sigma2=sigma2,
                    Pmax=Pmax,
                    subset_cache=subset_cache,
                )
                if not feas_opt:
                    infeas_opt += 1
                    powers_opt_all.append(Pmax)
                else:
                    powers_opt_all.append(total_power_opt)

                # Match legacy compare policy: random subset for ZF heuristic.
                idx_zf_np = rng.choice(K, size=U, replace=False).astype(np.int32)
                H_sub_tf = tf.gather(H_eff, tf.convert_to_tensor(idx_zf_np, dtype=tf.int32), axis=0)
                _, _, W_zf = _zf_heur_power_given_subset_tf(
                    H_sub=H_sub_tf,
                    gamma_lin=gamma_lin,
                    sigma2=sigma2,
                    Pmax=Pmax,
                )
                p_eval_zf, feas_eval_zf = evaluate_single_rb_with_p2_core(
                    cfg, H_sub_tf.numpy(), W_zf.numpy(), gamma_lin
                )
                if not feas_eval_zf:
                    infeas_zf += 1
                    powers_zf_all.append(Pmax)
                else:
                    powers_zf_all.append(p_eval_zf)

            done += bs
            chunk_id += 1

        results["gamma_dB"].append(gamma_dB)
        results["P2_opt_avg"].append(float(np.mean(powers_opt_all)) if powers_opt_all else np.nan)
        results["P2_zf_avg"].append(float(np.mean(powers_zf_all)) if powers_zf_all else np.nan)
        results["P2_opt_infeas"].append(infeas_opt)
        results["P2_zf_infeas"].append(infeas_zf)

        print(f"  -> P2_opt: meanP={results['P2_opt_avg'][-1]:.6f} W, infeasible={infeas_opt}/{N_mc}")
        print(f"  -> ZF-heur: meanP={results['P2_zf_avg'][-1]:.6f} W, infeasible={infeas_zf}/{N_mc}")

    return results


def save_results(results, out_dir="results", cfg: SystemConfig | None = None):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "results_p2_gpu.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "gamma_dB",
            "P2_opt_avg_power",
            "P2_zf_avg_power",
            "P2_opt_infeasible_count",
            "P2_zf_infeasible_count",
        ])
        for i in range(len(results["gamma_dB"])):
            w.writerow([
                results["gamma_dB"][i],
                results["P2_opt_avg"][i],
                results["P2_zf_avg"][i],
                results["P2_opt_infeas"][i],
                results["P2_zf_infeas"][i],
            ])
    print("Saved CSV:", csv_path)

    fig_path = os.path.join(out_dir, "fig_p2_gpu.png")
    gamma = np.array(results["gamma_dB"])
    p_opt = np.array(results["P2_opt_avg"])
    p_zf = np.array(results["P2_zf_avg"])

    plt.figure()
    plt.plot(gamma, p_opt, marker="o", label="P2 optimal (GPU exact-subset)")
    plt.plot(gamma, p_zf, marker="s", label="ZF heuristic (GPU)")
    plt.xlabel("Target SINR Γ (dB)")
    plt.ylabel("Average total transmit power (W)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()

    title = "P2 (GPU pipeline): Power minimization vs SINR target"
    if cfg is not None:
        U = min(cfg.U_max, cfg.K, cfg.M)
        title += f"\nM={cfg.M}, K={cfg.K}, U={U}, P_tot={cfg.P_tot:.1f} W"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved figure:", fig_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_mc", type=int, default=1000,
                        help="Number of Monte Carlo channel realizations")
    parser.add_argument("--gammas", type=str, default="-5,0,5,10,15,20",
                        help='List of target SINR in dB, e.g. "-5,0,5,10" or "-5 0 5 10"')
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed (overrides cfg.seed)")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Batch size for channel generation")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Output directory for CSV and figure")
    args = parser.parse_args()

    gpus = setup_gpu_memory_growth()
    print("GPUs detected:", [g.name for g in gpus])

    cfg = SystemConfig()
    gamma_list = parse_gamma_list(args.gammas)
    print("Using gamma_dB list:", gamma_list)
    print("SystemConfig:", cfg)

    results = run_p2_gpu(
        cfg,
        N_mc=args.N_mc,
        gamma_dB_list=gamma_list,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    save_results(results, out_dir=args.outdir, cfg=cfg)


if __name__ == "__main__":
    main()
