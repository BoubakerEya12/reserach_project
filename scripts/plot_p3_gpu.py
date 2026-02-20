import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import SystemConfig
from sim.sionna_channel import generate_h_rb_true, setup_gpu_memory_growth
from scripts.evaluate_p3 import evaluate_p3


def _median_last_dim(x: tf.Tensor) -> tf.Tensor:
    xs = tf.sort(x, axis=-1)
    k = tf.shape(xs)[-1]
    mid = k // 2
    hi = tf.gather(xs, mid, axis=-1)
    lo = tf.gather(xs, tf.maximum(mid - 1, 0), axis=-1)
    return tf.where(tf.equal(k % 2, 0), 0.5 * (lo + hi), hi)


def _beta_alpha_from_h(h_true: tf.Tensor, gamma: float = 0.5):
    # h_true: [B,K,N_RB,M]
    beta_raw = tf.reduce_mean(tf.abs(h_true) ** 2, axis=[2, 3])  # [B,K]
    beta_ref = tf.maximum(_median_last_dim(beta_raw), 1e-12)      # [B]
    beta = beta_raw / beta_ref[:, None]

    alpha_raw = tf.pow(tf.maximum(beta, 1e-15), tf.cast(gamma, beta.dtype))
    alpha = alpha_raw * (tf.cast(tf.shape(beta)[1], beta.dtype) / tf.reduce_sum(alpha_raw, axis=1, keepdims=True))
    return beta, alpha


def _gather_h_sel(h_true: tf.Tensor, sel: tf.Tensor) -> tf.Tensor:
    # h_true: [B,K,N_RB,M], sel:[B,N_RB,U]
    valid = sel >= 0
    sel_safe = tf.where(valid, sel, tf.zeros_like(sel))
    h_nbkm = tf.transpose(h_true, [0, 2, 1, 3])
    h_sel = tf.gather(h_nbkm, sel_safe, axis=2, batch_dims=2)  # [B,N_RB,U,M]
    h_sel = tf.where(valid[..., None], h_sel, tf.zeros_like(h_sel))
    return h_sel


def _zf_equal_power_beams(h_true: tf.Tensor, sel: tf.Tensor, p_rb_max: float) -> tf.Tensor:
    h_sel = _gather_h_sel(h_true, sel)                          # [B,N,U,M]
    b = tf.shape(h_sel)[0]
    n_rb = tf.shape(h_sel)[1]
    u = tf.shape(h_sel)[2]

    h_h = tf.math.conj(tf.transpose(h_sel, [0, 1, 3, 2]))       # [B,N,M,U]
    gram = tf.matmul(h_sel, h_h)                                 # [B,N,U,U]
    eye_u = tf.eye(u, batch_shape=[b, n_rb], dtype=gram.dtype)
    inv = tf.linalg.inv(gram + tf.cast(1e-6, gram.dtype) * eye_u)
    v = tf.matmul(h_h, inv)                                      # [B,N,M,U]

    col_norm = tf.sqrt(tf.reduce_sum(tf.abs(v) ** 2, axis=2, keepdims=True) + 1e-12)
    v = v / tf.cast(col_norm, v.dtype)

    p_each = tf.cast(p_rb_max, tf.float32) / tf.cast(u, tf.float32)
    w = v * tf.cast(tf.sqrt(tf.maximum(p_each, 0.0)), v.dtype)
    return w


def _wmmse_precoder_batched(h_sel: tf.Tensor,
                            alpha_sel: tf.Tensor,
                            p_rb_max: float,
                            sigma2: float,
                            n_iter: int = 20) -> tf.Tensor:
    # h_sel: [B,N,U,M], alpha_sel:[B,N,U]
    b = tf.shape(h_sel)[0]
    n_rb = tf.shape(h_sel)[1]
    u = tf.shape(h_sel)[2]
    m = tf.shape(h_sel)[3]
    s = b * n_rb

    h = tf.reshape(h_sel, [s, u, m])
    alpha = tf.reshape(alpha_sel, [s, u])

    # init: ZF + equal power
    h_h = tf.math.conj(tf.transpose(h, [0, 2, 1]))             # [S,M,U]
    gram = tf.matmul(h, h_h)                                    # [S,U,U]
    eye_u = tf.eye(u, batch_shape=[s], dtype=gram.dtype)
    v = tf.matmul(h_h, tf.linalg.inv(gram + tf.cast(1e-6, gram.dtype) * eye_u))
    v = v / tf.cast(tf.sqrt(tf.reduce_sum(tf.abs(v) ** 2, axis=1, keepdims=True) + 1e-12), v.dtype)
    f = v * tf.cast(tf.sqrt(tf.cast(p_rb_max, tf.float32) / tf.cast(u, tf.float32)), v.dtype)

    eye_m = tf.eye(m, batch_shape=[s], dtype=h.dtype)
    sigma2_t = tf.cast(sigma2, tf.float32)

    for _ in range(n_iter):
        hf = tf.matmul(h, f)                                    # [S,U,U]
        pwr = tf.abs(hf) ** 2
        den = sigma2_t + tf.reduce_sum(pwr, axis=-1)            # [S,U]
        sig = tf.linalg.diag_part(hf)                           # [S,U]

        ueq = sig / tf.cast(den + 1e-9, sig.dtype)              # [S,U]
        e = 1.0 - 2.0 * tf.math.real(ueq * tf.math.conj(sig)) + (tf.abs(ueq) ** 2) * den
        e = tf.maximum(e, 1e-9)
        wgt = alpha / e

        h_c = tf.math.conj(h)
        outer = tf.einsum("sum,sun->sumn", h_c, h)             # [S,U,M,M]
        coef = wgt * (tf.abs(ueq) ** 2)                         # [S,U]
        a = tf.reduce_sum(tf.cast(coef[:, :, None, None], outer.dtype) * outer, axis=1)
        a = a + tf.cast(1e-6, a.dtype) * eye_m

        bmat = tf.einsum(
            "su,su,sum->smu",
            tf.cast(wgt, h_c.dtype),
            tf.math.conj(ueq),
            h_c,
        )

        # lambda = 0 candidate
        f0 = tf.linalg.solve(a, bmat)
        p0 = tf.reduce_sum(tf.abs(f0) ** 2, axis=[1, 2])
        active = p0 > tf.cast(p_rb_max, p0.dtype)

        lam_lo = tf.zeros_like(p0)
        lam_hi = tf.ones_like(p0)

        # expand upper bound
        for _ in range(10):
            amat = a + tf.cast(lam_hi[:, None, None], a.dtype) * eye_m
            fh = tf.linalg.solve(amat, bmat)
            ph = tf.reduce_sum(tf.abs(fh) ** 2, axis=[1, 2])
            need = tf.logical_and(active, ph > tf.cast(p_rb_max, ph.dtype))
            lam_hi = tf.where(need, 2.0 * lam_hi, lam_hi)

        # bisection
        for _ in range(20):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            amat = a + tf.cast(lam_mid[:, None, None], a.dtype) * eye_m
            fm = tf.linalg.solve(amat, bmat)
            pm = tf.reduce_sum(tf.abs(fm) ** 2, axis=[1, 2])
            too_high = pm > tf.cast(p_rb_max, pm.dtype)
            lam_lo = tf.where(tf.logical_and(active, too_high), lam_mid, lam_lo)
            lam_hi = tf.where(tf.logical_and(active, tf.logical_not(too_high)), lam_mid, lam_hi)

        f_bis = tf.linalg.solve(a + tf.cast(lam_hi[:, None, None], a.dtype) * eye_m, bmat)
        f = tf.where(active[:, None, None], f_bis, f0)

    return tf.reshape(f, [b, n_rb, m, u])


def _sel_top_scores(scores_bk: tf.Tensor, n_rb: int, u: int) -> tf.Tensor:
    # scores_bk: [B,K] -> sel [B,N_RB,U]
    idx = tf.math.top_k(scores_bk, k=u, sorted=True).indices
    return tf.tile(idx[:, None, :], [1, n_rb, 1])


def _enforce_power_constraints(w: tf.Tensor, p_rb_budget: float, p_tot_budget: float) -> tf.Tensor:
    # w: [B,N_RB,M,U]
    p_rb = tf.reduce_sum(tf.abs(w) ** 2, axis=[2, 3])  # [B,N_RB]
    s_rb = tf.sqrt(
        tf.minimum(
            1.0,
            tf.cast(p_rb_budget, p_rb.dtype) / (p_rb + 1e-12),
        )
    )  # [B,N_RB]
    w = w * tf.cast(s_rb[:, :, None, None], w.dtype)

    p_tot = tf.reduce_sum(tf.abs(w) ** 2, axis=[1, 2, 3])  # [B]
    s_tot = tf.sqrt(
        tf.minimum(
            1.0,
            tf.cast(p_tot_budget, p_tot.dtype) / (p_tot + 1e-12),
        )
    )  # [B]
    w = w * tf.cast(s_tot[:, None, None, None], w.dtype)
    return w


def _method_wsr(h_true: tf.Tensor,
                cfg: SystemConfig,
                sigma2: float,
                mode: str,
                seed_i: int):
    b = tf.shape(h_true)[0]
    n_rb = int(cfg.N_RB)
    k = int(cfg.K)
    u = int(min(cfg.U_max, cfg.K, cfg.M))

    beta, alpha = _beta_alpha_from_h(h_true)

    # Enforce both C2 and C3 by using a per-RB budget that is compatible
    # with the total BS budget.
    p_rb_budget = float(min(cfg.P_RB_max, cfg.P_tot / max(cfg.N_RB, 1)))

    if mode == "zf_beta":
        sel = _sel_top_scores(beta, n_rb=n_rb, u=u)
        w = _zf_equal_power_beams(h_true, sel, p_rb_max=p_rb_budget)

    elif mode == "zf_random":
        rnd = tf.random.stateless_uniform([b, n_rb, k], seed=tf.constant([seed_i, 12345], tf.int32))
        sel = tf.math.top_k(rnd, k=u, sorted=False).indices
        w = _zf_equal_power_beams(h_true, sel, p_rb_max=p_rb_budget)

    elif mode == "wmmse_alpha_beta":
        scores = alpha * beta
        sel = _sel_top_scores(scores, n_rb=n_rb, u=u)
        h_sel = _gather_h_sel(h_true, sel)

        alpha_nbk = tf.broadcast_to(alpha[:, None, :], [b, n_rb, k])
        alpha_sel = tf.gather(alpha_nbk, sel, axis=2, batch_dims=2)
        w = _wmmse_precoder_batched(h_sel, alpha_sel, p_rb_max=p_rb_budget, sigma2=sigma2)

    elif mode == "mrt_single":
        u_eval = int(min(cfg.U_max, cfg.K, cfg.M))
        score = alpha * beta
        u_star = tf.argmax(score, axis=1, output_type=tf.int32)                 # [B]
        first = tf.tile(u_star[:, None, None], [1, n_rb, 1])                    # [B,N,1]
        if u_eval > 1:
            pad = -tf.ones([b, n_rb, u_eval - 1], dtype=tf.int32)
            sel = tf.concat([first, pad], axis=-1)
        else:
            sel = first

        h_nbkm = tf.transpose(h_true, [0, 2, 1, 3])
        h_star = tf.gather(h_nbkm, first, axis=2, batch_dims=2)[:, :, 0, :]     # [B,N,M]
        v = tf.math.conj(h_star)
        v = v / tf.cast(tf.sqrt(tf.reduce_sum(tf.abs(v) ** 2, axis=-1, keepdims=True) + 1e-12), v.dtype)

        w0 = tf.cast(tf.sqrt(tf.constant(p_rb_budget, tf.float32)), v.dtype) * v  # [B,N,M]
        w = w0[:, :, :, None]                                                   # [B,N,M,1]
        if u_eval > 1:
            w = tf.concat([w, tf.zeros([b, n_rb, tf.shape(w0)[2], u_eval - 1], dtype=w.dtype)], axis=-1)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    w = _enforce_power_constraints(w, p_rb_budget=p_rb_budget, p_tot_budget=float(cfg.P_tot))

    out = evaluate_p3(
        h_true=h_true,
        sel=tf.cast(sel, tf.int32),
        W=tf.cast(w, tf.complex64),
        alpha=tf.cast(alpha, tf.float32),
        sigma2=tf.constant(float(sigma2), tf.float32),
        P_RB_max=tf.constant(float(p_rb_budget), tf.float32),
        P_tot=tf.constant(float(cfg.P_tot), tf.float32),
        U_max=u,
        apply_channel_conjugate=False,
        power_tol=1e-4,
    )
    return out["objective_wsr"], out["feasible"]


def evaluate_p3_wsr_at_snr_gpu(cfg: SystemConfig,
                               snr_db: float,
                               n_slots: int,
                               seed: int,
                               mode: str,
                               chunk_size: int = 256):
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    p_rb_budget = float(min(cfg.P_RB_max, cfg.P_tot / max(cfg.N_RB, 1)))
    sigma2 = p_rb_budget / snr_lin

    sum_v = 0.0
    sum_v2 = 0.0
    count = 0
    infeas = 0

    done = 0
    chunk_id = 0
    while done < n_slots:
        b = min(chunk_size, n_slots - done)
        tf.random.set_seed(seed + 1000 * chunk_id)
        h_true = generate_h_rb_true(cfg, batch_size=b)                         # [B,K,N_RB,M]
        # Normalize global channel scale per sample to avoid near-zero rates
        # from raw 3GPP pathloss values while keeping relative user geometry.
        p_user = tf.reduce_mean(tf.abs(h_true) ** 2, axis=[2, 3])              # [B,K]
        p_ref = tf.maximum(_median_last_dim(p_user), 1e-12)                    # [B]
        h_true = h_true / tf.cast(tf.sqrt(p_ref)[:, None, None, None], h_true.dtype)

        vals, feas = _method_wsr(h_true, cfg=cfg, sigma2=sigma2, mode=mode, seed_i=seed + 37 * chunk_id)
        vals_np = vals.numpy().astype(np.float64)
        feas_np = feas.numpy()

        sum_v += float(np.sum(vals_np))
        sum_v2 += float(np.sum(vals_np ** 2))
        count += b
        infeas += int(np.sum(~feas_np))

        done += b
        chunk_id += 1

    mean = sum_v / max(count, 1)
    var = max(sum_v2 / max(count, 1) - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean, std, infeas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=30.0)
    parser.add_argument("--snr_step", type=float, default=5.0)
    parser.add_argument("--n_slots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--out_csv", type=str, default="results/p3_wsr_vs_snr_gpu.csv")
    parser.add_argument("--out_fig", type=str, default="results/p3_wsr_vs_snr_gpu.png")
    args = parser.parse_args()

    gpus = setup_gpu_memory_growth()
    print("GPUs detected:", [g.name for g in gpus])

    cfg = SystemConfig()
    snr_grid = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)

    print("=== P3 WSR vs SNR (GPU) ===")
    print(f"M={cfg.M}, K={cfg.K}, N_RB={cfg.N_RB}, U_max={cfg.U_max}")
    print(f"Per-RB power cap P_RB_max = {cfg.P_RB_max:.2f} W")
    print(f"SNR grid: {snr_grid} dB")
    print(f"Monte-Carlo: {args.n_slots} slots per point | chunk_size={args.chunk_size}\n")

    methods = [
        ("zf_beta", "ZF (greedy on beta)"),
        ("zf_random", "ZF (random users)"),
        ("wmmse_alpha_beta", "WMMSE (greedy on alpha·beta)"),
        ("mrt_single", "Single-user MRT (upper bound)"),
    ]

    n_methods = len(methods)
    mean_wsr = np.zeros((n_methods, len(snr_grid)), dtype=np.float64)
    std_wsr = np.zeros_like(mean_wsr)
    infeas = np.zeros_like(mean_wsr, dtype=np.int64)

    for i_snr, snr_db in enumerate(snr_grid):
        print(f"SNR = {snr_db:.1f} dB")
        for m, (mode, label) in enumerate(methods):
            mu, sig, inf = evaluate_p3_wsr_at_snr_gpu(
                cfg=cfg,
                snr_db=float(snr_db),
                n_slots=args.n_slots,
                seed=args.seed + i_snr * 100 + m,
                mode=mode,
                chunk_size=args.chunk_size,
            )
            mean_wsr[m, i_snr] = mu
            std_wsr[m, i_snr] = sig
            infeas[m, i_snr] = inf
            print(f"  {label:<30s} -> mean = {mu:9.3f}, std = {sig:9.3f}, infeas={inf}/{args.n_slots}")
        print()

    out_csv_dir = os.path.dirname(args.out_csv)
    if out_csv_dir:
        os.makedirs(out_csv_dir, exist_ok=True)

    header = "snr_db," + ",".join([f"{name}_mean,{name}_std,{name}_infeas" for name, _ in methods])
    rows = []
    for i in range(len(snr_grid)):
        row = [snr_grid[i]]
        for m in range(n_methods):
            row.extend([mean_wsr[m, i], std_wsr[m, i], float(infeas[m, i])])
        rows.append(row)
    np.savetxt(args.out_csv, np.asarray(rows, dtype=np.float64), delimiter=",", header=header, comments="")
    print(f"CSV saved to: {args.out_csv}")

    out_fig_dir = os.path.dirname(args.out_fig)
    if out_fig_dir:
        os.makedirs(out_fig_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    markers = ["o", "s", "^", "d"]
    linestyles = ["-", "--", "-.", ":"]
    for m, (_, label) in enumerate(methods):
        plt.errorbar(
            snr_grid,
            mean_wsr[m, :],
            yerr=std_wsr[m, :],
            marker=markers[m % len(markers)],
            linestyle=linestyles[m % len(linestyles)],
            capsize=3,
            label=label,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Weighted Sum Rate (bit/s/Hz)")
    plt.title("P3 - WSR vs SNR (GPU, Sionna UMa)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)
    print(f"Figure saved to: {args.out_fig}")


if __name__ == "__main__":
    main()
