# scripts/plot_p2.py
# ----------------------------------------------------------
# P2: Power minimization under SINR targets Γ (GPU)
# Methods aligned with plot_p2_compare.py:
#   - P2_opt (GPU surrogate): best of {ZF, RZF, MRT} fixed-directions solvers
#   - P2_zf  (GPU): ZF fixed-directions solver
# Uses evaluate_p2.py for objective/feasibility checks.
# ----------------------------------------------------------

import os
import re
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import SystemConfig
from sim.sionna_channel import generate_h_rb_true
from scripts.evaluate_p2 import evaluate_p2


def parse_gamma_list(s: str):
    parts = re.split(r"[,\s;]+", s.strip())
    return [float(x) for x in parts if x]


def _select_top_u_batch(H_eff: tf.Tensor, U: int) -> tf.Tensor:
    # H_eff: [B,K,M]
    norms = tf.math.real(tf.reduce_sum(H_eff * tf.math.conj(H_eff), axis=2))  # [B,K]
    _, idx = tf.math.top_k(norms, k=U, sorted=True)
    return tf.cast(idx, tf.int32)  # [B,U]


def _gather_H_sub(H_eff: tf.Tensor, idx: tf.Tensor) -> tf.Tensor:
    # H_eff: [B,K,M], idx: [B,U] -> H_sub: [B,U,M]
    B = tf.shape(H_eff)[0]
    b_idx = tf.tile(tf.range(B)[:, None], [1, tf.shape(idx)[1]])
    gather_idx = tf.stack([b_idx, idx], axis=-1)  # [B,U,2]
    return tf.gather_nd(H_eff, gather_idx)


def _directions_zf(H_sub: tf.Tensor, reg: float = 1e-6) -> tf.Tensor:
    # H_sub: [B,U,M] -> V:[B,M,U]
    B = tf.shape(H_sub)[0]
    U = tf.shape(H_sub)[1]
    Ht = tf.transpose(H_sub, [0, 2, 1])
    Gram = tf.matmul(tf.linalg.adjoint(Ht), Ht)
    eye = tf.eye(U, batch_shape=[B], dtype=Gram.dtype)
    V = tf.matmul(Ht, tf.linalg.inv(Gram + tf.cast(reg, Gram.dtype) * eye))
    V = V / (tf.norm(V, axis=1, keepdims=True) + 1e-12)
    return tf.cast(V, tf.complex64)


def _directions_rzf(H_sub: tf.Tensor, sigma2: tf.Tensor, Pmax: tf.Tensor, reg: float = 1e-6) -> tf.Tensor:
    B = tf.shape(H_sub)[0]
    U = tf.shape(H_sub)[1]
    Ht = tf.transpose(H_sub, [0, 2, 1])
    Gram = tf.matmul(tf.linalg.adjoint(Ht), Ht)
    alpha = tf.cast(U, tf.float32) * tf.cast(sigma2, tf.float32) / tf.maximum(tf.cast(Pmax, tf.float32), 1e-12)
    alpha = tf.cast(alpha, Gram.dtype)
    eye = tf.eye(U, batch_shape=[B], dtype=Gram.dtype)
    A = Gram + alpha * eye + tf.cast(reg, Gram.dtype) * eye
    V = tf.matmul(Ht, tf.linalg.inv(A))
    V = V / (tf.norm(V, axis=1, keepdims=True) + 1e-12)
    return tf.cast(V, tf.complex64)


def _directions_mrt(H_sub: tf.Tensor) -> tf.Tensor:
    V = tf.transpose(tf.math.conj(H_sub), [0, 2, 1])
    V = V / (tf.norm(V, axis=1, keepdims=True) + 1e-12)
    return tf.cast(V, tf.complex64)


def _solve_power_fixed_dirs(H_sub: tf.Tensor, V: tf.Tensor, gamma_lin: float, sigma2: tf.Tensor, Pmax: tf.Tensor):
    # H_sub: [B,U,M], V:[B,M,U]
    B = tf.shape(H_sub)[0]
    U = tf.shape(H_sub)[1]

    HV = tf.einsum("bum,bmv->buv", H_sub, V)    # [B,U,U]
    G = tf.abs(HV) ** 2
    gkk = tf.maximum(tf.linalg.diag_part(G), tf.constant(1e-12, G.dtype))

    gamma = tf.cast(gamma_lin, G.dtype)
    F = gamma * (G / gkk[:, :, None])
    F = F - tf.linalg.diag(tf.linalg.diag_part(F))
    u = gamma * tf.cast(sigma2, G.dtype) / gkk

    I = tf.eye(U, batch_shape=[B], dtype=G.dtype)
    p = tf.linalg.solve(I - F, u[..., None])[..., 0]
    p = tf.math.real(p)
    p = tf.maximum(p, 0.0)

    W = V * tf.cast(tf.sqrt(p[:, None, :]), V.dtype)  # [B,M,U]
    p_tot = tf.reduce_sum(p, axis=1)

    feasible = tf.reduce_all(tf.math.is_finite(p), axis=1) & (p_tot <= tf.cast(Pmax, p_tot.dtype) * 1.0001)
    return p, W, p_tot, feasible


def _evaluate_method(cfg: SystemConfig, H_eff: tf.Tensor, idx: tf.Tensor, W_method: tf.Tensor, gamma_lin: float):
    # Build tensors for evaluate_p2 over one RB
    B = tf.shape(H_eff)[0]
    K = tf.shape(H_eff)[1]
    U = tf.shape(idx)[1]

    h_true = H_eff[:, :, None, :]                 # [B,K,1,M]
    sel = idx[:, None, :]                         # [B,1,U]
    W = W_method[:, None, :, :]                   # [B,1,M,U]

    R_req = float(cfg.B_RB) * np.log2(1.0 + gamma_lin)
    R_th = tf.fill([K], tf.cast(R_req, tf.float32))

    out = evaluate_p2(
        h_true=h_true,
        sel=sel,
        W=W,
        R_th=R_th,
        sigma2=tf.constant(float(cfg.sigma2), tf.float32),
        B_RB=tf.constant(float(cfg.B_RB), tf.float32),
        P_tot=tf.constant(float(cfg.P_tot), tf.float32),
        U_max=int(cfg.U_max),
        rate_per_hz=False,
        require_all_users=False,
        r_max=1,
    )
    return out


def run_p2_gpu(cfg: SystemConfig, N_mc: int, gamma_dB_list, seed: int | None = None, chunk_size: int = 512):
    if seed is not None:
        tf.random.set_seed(seed)

    U = min(cfg.U_max, cfg.K, cfg.M)
    sigma2 = tf.constant(float(cfg.sigma2), tf.float32)
    Pmax = tf.constant(float(cfg.P_tot), tf.float32)
    rb_idx = int(getattr(cfg, "rb_index", 0))
    if rb_idx < 0 or rb_idx >= int(cfg.N_RB):
        rb_idx = 0

    # Running accumulators per gamma (avoid storing all samples in memory)
    sums_opt = {float(g): 0.0 for g in gamma_dB_list}
    sums_zf = {float(g): 0.0 for g in gamma_dB_list}
    infeas_opt = {float(g): 0 for g in gamma_dB_list}
    infeas_zf = {float(g): 0 for g in gamma_dB_list}

    done = 0
    while done < N_mc:
        bs = min(chunk_size, N_mc - done)
        h_rb_true = generate_h_rb_true(cfg, batch_size=bs)      # [bs,K,N_RB,M]
        H_eff = h_rb_true[:, :, rb_idx, :]                      # [bs,K,M]

        idx = _select_top_u_batch(H_eff, U)
        H_sub = _gather_H_sub(H_eff, idx)                       # [bs,U,M]

        for gamma_dB in gamma_dB_list:
            g = float(gamma_dB)
            gamma_lin = 10.0 ** (g / 10.0)

            V_zf = _directions_zf(H_sub)
            V_rzf = _directions_rzf(H_sub, sigma2=sigma2, Pmax=Pmax)
            V_mrt = _directions_mrt(H_sub)

            _, W_zf, _, _ = _solve_power_fixed_dirs(H_sub, V_zf, gamma_lin, sigma2, Pmax)
            _, W_rzf, _, _ = _solve_power_fixed_dirs(H_sub, V_rzf, gamma_lin, sigma2, Pmax)
            _, W_mrt, _, _ = _solve_power_fixed_dirs(H_sub, V_mrt, gamma_lin, sigma2, Pmax)

            out_zf = _evaluate_method(cfg, H_eff, idx, W_zf, gamma_lin)
            out_rzf = _evaluate_method(cfg, H_eff, idx, W_rzf, gamma_lin)
            out_mrt = _evaluate_method(cfg, H_eff, idx, W_mrt, gamma_lin)

            feas_zf = out_zf["feasible"]
            feas_rzf = out_rzf["feasible"]
            feas_mrt = out_mrt["feasible"]

            pz = tf.where(feas_zf, out_zf["objective_power"], tf.cast(Pmax, out_zf["objective_power"].dtype))
            pr = tf.where(feas_rzf, out_rzf["objective_power"], tf.cast(Pmax, out_rzf["objective_power"].dtype))
            pm = tf.where(feas_mrt, out_mrt["objective_power"], tf.cast(Pmax, out_mrt["objective_power"].dtype))

            Pstack = tf.stack([pz, pr, pm], axis=1)  # [bs,3]
            pbest = tf.reduce_min(Pstack, axis=1)

            sums_opt[g] += float(tf.reduce_sum(pbest).numpy())
            sums_zf[g] += float(tf.reduce_sum(pz).numpy())
            infeas_opt[g] += int(tf.reduce_sum(tf.cast(~(feas_zf | feas_rzf | feas_mrt), tf.int32)).numpy())
            infeas_zf[g] += int(tf.reduce_sum(tf.cast(~feas_zf, tf.int32)).numpy())

        done += bs
        print(f"Processed {done}/{N_mc} samples")

    results = {
        "gamma_dB": [],
        "P2_opt_avg": [],
        "P2_zf_avg": [],
        "P2_opt_infeas": [],
        "P2_zf_infeas": [],
    }
    for gamma_dB in gamma_dB_list:
        g = float(gamma_dB)
        results["gamma_dB"].append(g)
        results["P2_opt_avg"].append(sums_opt[g] / float(N_mc))
        results["P2_zf_avg"].append(sums_zf[g] / float(N_mc))
        results["P2_opt_infeas"].append(infeas_opt[g])
        results["P2_zf_infeas"].append(infeas_zf[g])
        print(
            f"Γ={g:.1f} dB -> "
            f"P2_opt={results['P2_opt_avg'][-1]:.6f} W | "
            f"P2_zf={results['P2_zf_avg'][-1]:.6f} W"
        )

    return results


def save_results(results, out_dir="results", cfg: SystemConfig | None = None):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "results_p2_compare_gpu.csv")
    with open(csv_path, "w", newline="") as f:
        import csv
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

    fig_path = os.path.join(out_dir, "fig_p2_compare_gpu.png")
    gamma = np.array(results["gamma_dB"])
    pbest = np.array(results["P2_opt_avg"])
    pzf = np.array(results["P2_zf_avg"])

    plt.figure(figsize=(8, 5.5))
    plt.plot(gamma, pbest, marker="o", label="P2 optimal (GPU surrogate)")
    plt.plot(gamma, pzf, marker="s", label="ZF heuristic (GPU)")
    plt.xlabel("Target SINR Γ (dB)")
    plt.ylabel("Average total transmit power (W)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    title = "P2 (GPU): Power minimization vs SINR target"
    if cfg is not None:
        title += f"\nM={cfg.M}, K={cfg.K}, U={min(cfg.U_max,cfg.K,cfg.M)}, P_tot={cfg.P_tot:.1f} W"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved figure:", fig_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_mc", type=int, default=1000)
    parser.add_argument("--gammas", type=str, default="-5,0,5,10,15,20")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    cfg = SystemConfig()
    gamma_list = parse_gamma_list(args.gammas)

    print("Using gamma_dB list:", gamma_list)
    print("SystemConfig:", cfg)

    results = run_p2_gpu(
        cfg, N_mc=args.N_mc, gamma_dB_list=gamma_list, seed=args.seed, chunk_size=args.chunk_size
    )
    save_results(results, out_dir=args.outdir, cfg=cfg)


if __name__ == "__main__":
    main()
