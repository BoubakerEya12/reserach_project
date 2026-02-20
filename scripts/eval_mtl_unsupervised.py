from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import tensorflow as tf

try:
    from config import SystemConfig
    from sim.mtl_unsupervised import MTLUnsupervisedModel
except ImportError:
    import sys

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from config import SystemConfig
    from sim.mtl_unsupervised import MTLUnsupervisedModel

try:
    from sim.sionna_channel import generate_channels
    HAS_SIONNA = True
except Exception:
    HAS_SIONNA = False


def _complex_normal(shape):
    re = tf.random.normal(shape, dtype=tf.float32)
    im = tf.random.normal(shape, dtype=tf.float32)
    return tf.complex(re, im) / tf.cast(tf.sqrt(2.0), tf.complex64)


def _fallback_channels(cfg: SystemConfig, batch_size: int):
    h_true = _complex_normal([batch_size, cfg.K, cfg.N_RB, cfg.M])
    eta = float(cfg.eta)
    e = _complex_normal([batch_size, cfg.K, cfg.N_RB, cfg.M])
    h_hat = tf.cast(eta, tf.complex64) * h_true + tf.cast(np.sqrt(max(1.0 - eta * eta, 0.0)), tf.complex64) * e
    return h_true, h_hat


def _hard_schedule_feasible(q_all: tf.Tensor, n_rb: int, u_max: int) -> tf.Tensor:
    """
    Convert soft schedule to hard schedule while enforcing per-RB capacity:
      - each user assigned to at most one RB
      - each RB serves at most u_max users

    Greedy assignment:
      - users sorted by confidence
      - each user placed on the best RB with available capacity
      - if no RB has capacity, user remains unscheduled
    """
    q_np = q_all.numpy()  # [B,K,N_RB+1]
    bsz, k_users, _ = q_np.shape
    out = np.zeros((bsz, k_users, n_rb), dtype=np.float32)

    for b in range(bsz):
        probs_rb = q_np[b, :, :n_rb]  # [K,N_RB]
        best_prob = np.max(probs_rb, axis=1)  # [K]
        user_order = np.argsort(-best_prob)   # descending confidence

        load = np.zeros((n_rb,), dtype=np.int32)
        for u in user_order:
            rb_order = np.argsort(-probs_rb[u])  # best RB first
            placed = False
            for n in rb_order:
                if load[n] < int(u_max):
                    out[b, u, n] = 1.0
                    load[n] += 1
                    placed = True
                    break
            if not placed:
                # Keep unscheduled if all RBs are full.
                pass

    return tf.convert_to_tensor(out, dtype=tf.float32)


def _sum_rate_and_power(
    h_true: tf.Tensor,
    v_re: tf.Tensor,
    v_im: tf.Tensor,
    s: tf.Tensor,
    sigma2: float,
    b_rb: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Returns:
      sum_rate: [B]
      p_tot: [B]
      sinr_kn: [B,K,N_RB]
      rates_kn: [B,K,N_RB]
    """
    h_bnkm = tf.transpose(h_true, [0, 2, 1, 3])  # [B,N,K,M]
    v_re_bnkm = tf.transpose(v_re, [0, 2, 1, 3])  # [B,N,K,M]
    v_im_bnkm = tf.transpose(v_im, [0, 2, 1, 3])  # [B,N,K,M]
    s_bnk = tf.transpose(s, [0, 2, 1])  # [B,N,K]

    h_re = tf.math.real(h_bnkm)
    h_im = tf.math.imag(h_bnkm)

    re_proj = tf.einsum("bnkm,bnum->bnku", h_re, v_re_bnkm) + tf.einsum("bnkm,bnum->bnku", h_im, v_im_bnkm)
    im_proj = tf.einsum("bnkm,bnum->bnku", h_re, v_im_bnkm) - tf.einsum("bnkm,bnum->bnku", h_im, v_re_bnkm)
    gain = re_proj * re_proj + im_proj * im_proj

    desired = tf.linalg.diag_part(gain) * s_bnk
    total_rx = tf.einsum("bnu,bnku->bnk", s_bnk, gain)
    interf = tf.maximum(total_rx - desired, 0.0)
    sinr = desired / (interf + tf.convert_to_tensor(sigma2, dtype=desired.dtype) + 1e-9)
    rates = tf.cast(b_rb, sinr.dtype) * tf.math.log(1.0 + sinr) / tf.math.log(2.0)

    sum_rate = tf.reduce_sum(rates, axis=[1, 2])  # [B]
    p_tot = tf.reduce_sum(s, axis=[1, 2])  # [B]
    sinr_kn = tf.transpose(sinr, [0, 2, 1])
    rates_kn = tf.transpose(rates, [0, 2, 1])
    return sum_rate, p_tot, sinr_kn, rates_kn


def _constraint_stats(
    q_hard: tf.Tensor,
    s_hard: tf.Tensor,
    cfg: SystemConfig,
) -> Dict[str, float]:
    """
    Hard-constraint statistics:
      - R_max, U_max, P_RB_max, P_tot
    """
    assign_user = tf.reduce_sum(q_hard, axis=-1)  # [B,K]
    load_rb = tf.reduce_sum(q_hard, axis=1)  # [B,N_RB]

    c_rmax = tf.reduce_all(assign_user <= float(cfg.R_max), axis=1)
    c_user_exact = tf.reduce_all(tf.equal(assign_user, 1.0), axis=1)
    c_umax = tf.reduce_all(load_rb <= float(min(cfg.U_max, cfg.M)), axis=1)

    # Power constraints evaluated on effective hard allocation.
    p_rb = tf.reduce_sum(s_hard, axis=1)  # [B,N_RB]
    p_tot = tf.reduce_sum(s_hard, axis=[1, 2])  # [B]
    c_prb = tf.reduce_all(p_rb <= float(cfg.P_RB_max) + 1e-6, axis=1)
    c_ptot = p_tot <= float(cfg.P_tot) + 1e-6

    feasible_all = c_rmax & c_umax & c_prb & c_ptot

    return {
        "frac_rmax_ok": float(tf.reduce_mean(tf.cast(c_rmax, tf.float32)).numpy()),
        "frac_user_exact1": float(tf.reduce_mean(tf.cast(c_user_exact, tf.float32)).numpy()),
        "frac_umax_ok": float(tf.reduce_mean(tf.cast(c_umax, tf.float32)).numpy()),
        "frac_prb_ok": float(tf.reduce_mean(tf.cast(c_prb, tf.float32)).numpy()),
        "frac_ptot_ok": float(tf.reduce_mean(tf.cast(c_ptot, tf.float32)).numpy()),
        "frac_all_ok": float(tf.reduce_mean(tf.cast(feasible_all, tf.float32)).numpy()),
    }


def _save_summary_csv(path: str, summary: Dict[str, float]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])


def _save_samples_csv(path: str, data: Dict[str, np.ndarray]) -> None:
    keys = list(data.keys())
    n = len(data[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([float(data[k][i]) for k in keys])


def _save_plots(out_dir: str, sample_soft_rate: np.ndarray, sample_hard_rate: np.ndarray, sample_power: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.8, 4.5))
    plt.hist(sample_soft_rate, bins=40, alpha=0.6, label="soft")
    plt.hist(sample_hard_rate, bins=40, alpha=0.6, label="hard")
    plt.xlabel("Sum rate per sample (bps)")
    plt.ylabel("Count")
    plt.title("MTL Unsupervised: Sum-rate distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mtl_eval_sumrate_hist.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(6.8, 4.5))
    plt.hist(sample_power, bins=40, alpha=0.8)
    plt.xlabel("Total allocated power per sample (W)")
    plt.ylabel("Count")
    plt.title("MTL Unsupervised: Total Power distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mtl_eval_power_hist.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(6.8, 4.5))
    plt.scatter(sample_power, sample_hard_rate, s=8, alpha=0.45)
    plt.xlabel("Total power (W)")
    plt.ylabel("Hard sum-rate (bps)")
    plt.title("MTL Unsupervised: Power vs Sum-rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mtl_eval_power_vs_rate.png"), dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="results/mtl_unsupervised.weights.h5")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--save_dir", type=str, default="results")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--gamma_qos_db", type=float, default=5.0)
    args = ap.parse_args()

    cfg = SystemConfig()
    os.makedirs(args.save_dir, exist_ok=True)

    model = MTLUnsupervisedModel(
        K=cfg.K,
        N_RB=cfg.N_RB,
        M=cfg.M,
        U_max=min(cfg.U_max, cfg.M),
        P_RB_max=cfg.P_RB_max,
        P_tot=cfg.P_tot,
    )
    dummy = tf.complex(
        tf.zeros([1, cfg.K, cfg.N_RB, cfg.M], tf.float32),
        tf.zeros([1, cfg.K, cfg.N_RB, cfg.M], tf.float32),
    )
    _ = model(dummy, training=False)
    model.load_weights(args.weights)

    print("==== MTL Evaluation ====")
    print(f"Weights: {args.weights}")
    print(f"Sionna available: {HAS_SIONNA}")
    print(f"steps={args.steps}, batch_size={args.batch_size}")
    print(f"QoS target: gamma={args.gamma_qos_db:.2f} dB")

    all_soft_rate: List[np.ndarray] = []
    all_hard_rate: List[np.ndarray] = []
    all_power: List[np.ndarray] = []
    all_assign_mean: List[np.ndarray] = []
    all_load_mean: List[np.ndarray] = []
    all_user_rate_ok_frac: List[np.ndarray] = []
    all_user_sinr_ok_frac: List[np.ndarray] = []
    all_sample_all_rate_ok: List[np.ndarray] = []
    all_sample_all_sinr_ok: List[np.ndarray] = []

    constraint_accum = {
        "frac_rmax_ok": [],
        "frac_user_exact1": [],
        "frac_umax_ok": [],
        "frac_prb_ok": [],
        "frac_ptot_ok": [],
        "frac_all_ok": [],
    }

    for _ in range(args.steps):
        if HAS_SIONNA:
            h_true, h_hat = generate_channels(cfg, args.batch_size)
        else:
            h_true, h_hat = _fallback_channels(cfg, args.batch_size)

        h_true = tf.cast(h_true, tf.complex64)
        h_hat = tf.cast(h_hat, tf.complex64)
        out = model(h_hat, training=False)

        q = out["q"]
        q_all = out["q_all"]
        p = out["p"]
        v_re = out["v_re"]
        v_im = out["v_im"]

        s_soft = out["s"]
        q_hard = _hard_schedule_feasible(q_all, cfg.N_RB, min(cfg.U_max, cfg.M))
        s_hard = p * q_hard

        soft_rate, p_tot, _, _ = _sum_rate_and_power(h_true, v_re, v_im, s_soft, cfg.sigma2, cfg.B_RB)
        hard_rate, _, hard_sinr_kn, hard_rates_kn = _sum_rate_and_power(h_true, v_re, v_im, s_hard, cfg.sigma2, cfg.B_RB)

        assign_user = tf.reduce_sum(q_hard, axis=-1)  # [B,K]
        load_rb = tf.reduce_sum(q_hard, axis=1)  # [B,N_RB]
        stats = _constraint_stats(q_hard, s_hard, cfg)
        gamma_qos_lin = float(10.0 ** (args.gamma_qos_db / 10.0))
        rate_qos = float(cfg.B_RB * np.log2(1.0 + gamma_qos_lin))

        # User-level QoS on hard allocation.
        # R_max=1 => each user is scheduled on at most one RB, but we keep generic reductions.
        user_rates = tf.reduce_sum(hard_rates_kn, axis=-1)  # [B,K]
        user_sinr_sched = tf.reduce_sum(hard_sinr_kn * q_hard, axis=-1)  # [B,K]

        user_rate_ok = tf.cast(user_rates >= rate_qos, tf.float32)
        user_sinr_ok = tf.cast(user_sinr_sched >= gamma_qos_lin, tf.float32)
        sample_all_rate_ok = tf.cast(tf.reduce_all(user_rates >= rate_qos, axis=1), tf.float32)
        sample_all_sinr_ok = tf.cast(tf.reduce_all(user_sinr_sched >= gamma_qos_lin, axis=1), tf.float32)

        all_soft_rate.append(soft_rate.numpy())
        all_hard_rate.append(hard_rate.numpy())
        all_power.append(p_tot.numpy())
        all_assign_mean.append(tf.reduce_mean(assign_user, axis=1).numpy())
        all_load_mean.append(tf.reduce_mean(load_rb, axis=1).numpy())
        all_user_rate_ok_frac.append(tf.reduce_mean(user_rate_ok, axis=1).numpy())
        all_user_sinr_ok_frac.append(tf.reduce_mean(user_sinr_ok, axis=1).numpy())
        all_sample_all_rate_ok.append(sample_all_rate_ok.numpy())
        all_sample_all_sinr_ok.append(sample_all_sinr_ok.numpy())

        for k in constraint_accum:
            constraint_accum[k].append(stats[k])

    sample_soft_rate = np.concatenate(all_soft_rate, axis=0)
    sample_hard_rate = np.concatenate(all_hard_rate, axis=0)
    sample_power = np.concatenate(all_power, axis=0)
    sample_assign_mean = np.concatenate(all_assign_mean, axis=0)
    sample_load_mean = np.concatenate(all_load_mean, axis=0)
    sample_user_rate_ok_frac = np.concatenate(all_user_rate_ok_frac, axis=0)
    sample_user_sinr_ok_frac = np.concatenate(all_user_sinr_ok_frac, axis=0)
    sample_all_rate_ok = np.concatenate(all_sample_all_rate_ok, axis=0)
    sample_all_sinr_ok = np.concatenate(all_sample_all_sinr_ok, axis=0)

    summary = {
        "mean_soft_sum_rate_bps": float(np.mean(sample_soft_rate)),
        "mean_hard_sum_rate_bps": float(np.mean(sample_hard_rate)),
        "std_hard_sum_rate_bps": float(np.std(sample_hard_rate)),
        "mean_total_power_w": float(np.mean(sample_power)),
        "std_total_power_w": float(np.std(sample_power)),
        "mean_assign_per_user": float(np.mean(sample_assign_mean)),
        "mean_users_per_rb": float(np.mean(sample_load_mean)),
        "qos_gamma_db": float(args.gamma_qos_db),
        "qos_user_rate_ok_frac": float(np.mean(sample_user_rate_ok_frac)),
        "qos_user_sinr_ok_frac": float(np.mean(sample_user_sinr_ok_frac)),
        "qos_sample_all_users_rate_ok_frac": float(np.mean(sample_all_rate_ok)),
        "qos_sample_all_users_sinr_ok_frac": float(np.mean(sample_all_sinr_ok)),
        "frac_rmax_ok": float(np.mean(constraint_accum["frac_rmax_ok"])),
        "frac_user_exact1": float(np.mean(constraint_accum["frac_user_exact1"])),
        "frac_umax_ok": float(np.mean(constraint_accum["frac_umax_ok"])),
        "frac_prb_ok": float(np.mean(constraint_accum["frac_prb_ok"])),
        "frac_ptot_ok": float(np.mean(constraint_accum["frac_ptot_ok"])),
        "frac_all_ok": float(np.mean(constraint_accum["frac_all_ok"])),
    }

    summary_csv = os.path.join(args.save_dir, "mtl_eval_summary.csv")
    samples_csv = os.path.join(args.save_dir, "mtl_eval_samples.csv")
    _save_summary_csv(summary_csv, summary)
    _save_samples_csv(
        samples_csv,
        {
            "soft_sum_rate_bps": sample_soft_rate,
            "hard_sum_rate_bps": sample_hard_rate,
            "total_power_w": sample_power,
            "assign_per_user": sample_assign_mean,
            "users_per_rb": sample_load_mean,
            "user_rate_ok_frac": sample_user_rate_ok_frac,
            "user_sinr_ok_frac": sample_user_sinr_ok_frac,
            "sample_all_users_rate_ok": sample_all_rate_ok,
            "sample_all_users_sinr_ok": sample_all_sinr_ok,
        },
    )

    if not args.no_plots:
        _save_plots(args.save_dir, sample_soft_rate, sample_hard_rate, sample_power)

    print("Saved:", summary_csv)
    print("Saved:", samples_csv)
    if not args.no_plots:
        print("Saved plots:")
        print("-", os.path.join(args.save_dir, "mtl_eval_sumrate_hist.png"))
        print("-", os.path.join(args.save_dir, "mtl_eval_power_hist.png"))
        print("-", os.path.join(args.save_dir, "mtl_eval_power_vs_rate.png"))

    print("---- Summary ----")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
