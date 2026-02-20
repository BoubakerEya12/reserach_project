# scripts/plot_fig5_gpu_avec_optimal.py
# ---------------------------------------------------------------------
# P1: Balanced-SINR maximization (GPU-friendly) WITH Optimal curve (CPU)
# Methods: Optimal (teacher, CPU), ZF (equal power), RZF (equal power), optional Exhaustive (CPU)
# Uses TensorFlow ops for SINR evaluation (evaluate_p1).
# ---------------------------------------------------------------------

import os
import csv
import argparse
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import SystemConfig
from sim.sionna_channel import generate_h_rb_true
from scripts.evaluate_p1 import evaluate_p1
from sim.exhaustive_teacher import best_group_by_balanced_sinr


# =============================== Helpers ===============================

def abspath_here(path: str) -> str:
    return os.path.abspath(path)

def ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)

def assert_saved(path: str, kind: str = "file"):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else -1
    print(f"[SAVE-CHECK] {kind}: {abspath_here(path)} | exists={exists} | size={size} bytes")
    if (not exists) or (size <= 0):
        raise RuntimeError(f"[SAVE-CHECK] Failed to save {kind}: {abspath_here(path)}")

def lin2db(x):
    x = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(x, 1e-12))

def db2lin(db):
    return 10.0 ** (db / 10.0)

def clip_sinr(x_lin, max_db=60.0):
    return min(float(x_lin), 10.0 ** (max_db / 10.0))

def complex_normal_tf(shape):
    re = tf.random.normal(shape, dtype=tf.float32)
    im = tf.random.normal(shape, dtype=tf.float32)
    return tf.complex(re, im) / tf.cast(tf.sqrt(2.0), tf.complex64)


def min_sinr_selected_tf(HkM, sel, W_sel, sigma2):
    h_true = tf.convert_to_tensor(HkM)[None, :, None, :]
    sel_t = tf.convert_to_tensor(sel, dtype=tf.int32)[None, None, :]
    W_t = tf.convert_to_tensor(W_sel)[None, None, :, :]
    rho = tf.ones([tf.shape(h_true)[1]], dtype=tf.float32)

    out = evaluate_p1(
        h_true=h_true,
        sel=sel_t,
        W=W_t,
        rho=rho,
        sigma2=tf.constant(float(sigma2), tf.float32),
        P_RB_max=tf.constant(1e30, tf.float32),
        P_tot=tf.constant(1e30, tf.float32),
    )
    return float(out["min_sinr_ratio_rb"][0, 0].numpy())


def pick_subset_topU_by_norm_tf(HkM, U):
    norms = tf.math.real(tf.reduce_sum(HkM * tf.math.conj(HkM), axis=1))
    vals, idx = tf.math.top_k(norms, k=U, sorted=True)
    return tf.cast(idx, tf.int32)


def rzf_directions_tf(H_sel, sigma2, P_rb):
    H = tf.transpose(H_sel)  # [M,U]
    Gram = tf.matmul(tf.linalg.adjoint(H), H)  # [U,U]
    alpha = tf.cast(H_sel.shape[0], tf.float32) * sigma2 / tf.maximum(P_rb, 1e-12)
    reg = tf.cast(1e-6, Gram.dtype)
    A = Gram + tf.cast(alpha, Gram.dtype) * tf.eye(H_sel.shape[0], dtype=Gram.dtype) + reg * tf.eye(H_sel.shape[0], dtype=Gram.dtype)
    V = tf.matmul(H, tf.linalg.inv(A))  # [M,U]
    V = V / (tf.norm(V, axis=0, keepdims=True) + 1e-12)
    return tf.cast(V, tf.complex64)


def build_equal_power_beams_tf(H_sel, sigma2, P_rb, mode="rzf"):
    U = H_sel.shape[0]
    if mode == "zf":
        H = tf.transpose(H_sel)  # [M,U]
        Gram = tf.matmul(tf.linalg.adjoint(H), H)
        reg = tf.cast(1e-6, Gram.dtype)
        Gram_reg = Gram + reg * tf.eye(U, dtype=Gram.dtype)
        V = tf.matmul(H, tf.linalg.inv(Gram_reg))
        V = V / (tf.norm(V, axis=0, keepdims=True) + 1e-12)
    else:
        V = rzf_directions_tf(H_sel, sigma2, P_rb)
    p = tf.fill([U], tf.cast(P_rb, tf.float32) / max(U, 1))
    W = V * tf.cast(tf.sqrt(p[None, :]), V.dtype)
    return tf.cast(V, tf.complex64), tf.cast(p, tf.float32), tf.cast(W, tf.complex64)


def zf_equal_power_value_tf(HkM, sigma2, P_rb, U):
    sel = pick_subset_topU_by_norm_tf(HkM, U)
    H_sel = tf.gather(HkM, sel, axis=0)
    V, p, W = build_equal_power_beams_tf(H_sel, sigma2, P_rb, mode="zf")
    min_lin = min_sinr_selected_tf(HkM, sel, W, sigma2)
    return max(min_lin, 1e-30)


def rzf_equal_power_value_tf(HkM, sigma2, P_rb, U):
    sel = pick_subset_topU_by_norm_tf(HkM, U)
    H_sel = tf.gather(HkM, sel, axis=0)
    V, p, W = build_equal_power_beams_tf(H_sel, sigma2, P_rb, mode="rzf")
    min_lin = min_sinr_selected_tf(HkM, sel, W, sigma2)
    return max(min_lin, 1e-30)


def exhaustive_equal_power_value_cpu(HkM, sigma2, P_rb, U_max, mode="rzf"):
    K, M = HkM.shape
    best = -np.inf
    users = list(range(K))
    for U in range(1, min(U_max, K) + 1):
        for comb in itertools.combinations(users, U):
            sel = np.array(comb, dtype=int)
            H_sel = HkM[sel, :]
            if mode == "zf":
                H = H_sel.T
                Gram = H.conj().T @ H
                V = H @ np.linalg.pinv(Gram)
                V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
            else:
                H = H_sel.T
                Gram = H.conj().T @ H
                alpha = (U * sigma2 / max(P_rb, 1e-12))
                A = Gram + alpha * np.eye(U, dtype=Gram.dtype)
                V = H @ np.linalg.pinv(A)
                V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
            p = np.full(U, P_rb / max(U, 1), dtype=np.float64)
            W = V * np.sqrt(p.reshape(1, -1))
            min_lin = min_sinr_selected_tf(HkM, sel, W, sigma2)
            if min_lin > best:
                best = min_lin
    if not np.isfinite(best):
        return 1e-30
    return max(best, 1e-30)


# ========================== Monte-Carlo core ===========================

def run_case(cfg: SystemConfig,
             samples: int,
             snr_grid_db,
             include_large: bool,
             seed: int | None = None,
             verbose: bool = False,
             include_exhaustive: bool = False):
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)

    K = cfg.K
    M = cfg.M
    sigma2 = float(cfg.sigma2)
    U = min(cfg.U_max, K, M)

    ys_a_opt, ys_a_zf, ys_a_rzf, ys_a_ex = [], [], [], []
    ys_b_opt, ys_b_zf, ys_b_rzf, ys_b_ex = [], [], [], []

    # Sionna large-scale+small-scale
    HkM_eff_samples = None
    if include_large:
        if seed is not None:
            tf.random.set_seed(seed)
        h_rb_true = generate_h_rb_true(cfg, batch_size=samples)  # [B,K,N_RB,M]
        rb_idx = int(getattr(cfg, "rb_index", 0))
        if rb_idx < 0 or rb_idx >= int(cfg.N_RB):
            rb_idx = 0
        HkM_eff_samples = h_rb_true[:, :, rb_idx, :]

    for snr_db in snr_grid_db:
        P_rb = sigma2 * db2lin(snr_db)

        acc_a = {"opt": 0.0, "zf": 0.0, "rzf": 0.0, "ex": 0.0}
        cnt_a = {"opt": 0,   "zf": 0,   "rzf": 0,   "ex": 0}

        acc_b = {"opt": 0.0, "zf": 0.0, "rzf": 0.0, "ex": 0.0}
        cnt_b = {"opt": 0,   "zf": 0,   "rzf": 0,   "ex": 0}

        if verbose:
            print(f"[P1-GPU] SNR={snr_db} dB, sample 0/{samples}")

        for s in range(samples):
            if verbose and (s % max(1, samples // 5) == 0) and s > 0:
                print(f"[P1-GPU] SNR={snr_db} dB, sample {s}/{samples}")

            # ---------- Small-scale only ----------
            Hs = complex_normal_tf([K, M])
            HkM = Hs

            # Optimal (CPU teacher)
            sel_o, W_o, p_o, sinr_o, score_lin0 = best_group_by_balanced_sinr(
                Hn=HkM.numpy().T,
                sigma2=sigma2,
                P_rb=P_rb,
                U_max=U,
            )
            if sel_o is not None and sel_o.size > 0:
                # Keep teacher objective exactly as in legacy CPU pipeline.
                opt_lin = float(np.min(sinr_o[sel_o]))
            else:
                opt_lin = float(score_lin0)

            zf_lin = zf_equal_power_value_tf(HkM, sigma2, P_rb, U)
            rzf_lin = rzf_equal_power_value_tf(HkM, sigma2, P_rb, U)
            ex_lin = None
            if include_exhaustive:
                ex_lin = exhaustive_equal_power_value_cpu(HkM.numpy(), sigma2, P_rb, U, mode="rzf")

            opt_lin = clip_sinr(opt_lin)
            zf_lin = clip_sinr(zf_lin)
            rzf_lin = clip_sinr(rzf_lin)

            acc_a["opt"] += opt_lin; cnt_a["opt"] += 1
            acc_a["zf"] += zf_lin;  cnt_a["zf"] += 1
            acc_a["rzf"] += rzf_lin; cnt_a["rzf"] += 1
            if include_exhaustive:
                ex_lin = clip_sinr(ex_lin)
                acc_a["ex"] += ex_lin; cnt_a["ex"] += 1

            # ---------- Large-scale + small-scale (Sionna) ----------
            if include_large:
                HkM_eff = HkM_eff_samples[s]

                sel_b, W_b, p_b, sinr_b, score_lin_b = best_group_by_balanced_sinr(
                    Hn=HkM_eff.numpy().T,
                    sigma2=sigma2,
                    P_rb=P_rb,
                    U_max=U,
                )
                if sel_b is not None and sel_b.size > 0:
                    # Keep teacher objective exactly as in legacy CPU pipeline.
                    opt_lin_b = float(np.min(sinr_b[sel_b]))
                else:
                    opt_lin_b = float(score_lin_b)

                zf_lin_b = zf_equal_power_value_tf(HkM_eff, sigma2, P_rb, U)
                rzf_lin_b = rzf_equal_power_value_tf(HkM_eff, sigma2, P_rb, U)
                ex_lin_b = None
                if include_exhaustive:
                    ex_lin_b = exhaustive_equal_power_value_cpu(HkM_eff.numpy(), sigma2, P_rb, U, mode="rzf")

                opt_lin_b = clip_sinr(opt_lin_b)
                zf_lin_b = clip_sinr(zf_lin_b)
                rzf_lin_b = clip_sinr(rzf_lin_b)

                acc_b["opt"] += opt_lin_b; cnt_b["opt"] += 1
                acc_b["zf"] += zf_lin_b;  cnt_b["zf"] += 1
                acc_b["rzf"] += rzf_lin_b; cnt_b["rzf"] += 1
                if include_exhaustive:
                    ex_lin_b = clip_sinr(ex_lin_b)
                    acc_b["ex"] += ex_lin_b; cnt_b["ex"] += 1

        ys_a_opt.append(lin2db(acc_a["opt"] / max(1, cnt_a["opt"])) )
        ys_a_zf.append(lin2db(acc_a["zf"] / max(1, cnt_a["zf"])) )
        ys_a_rzf.append(lin2db(acc_a["rzf"] / max(1, cnt_a["rzf"])) )
        if include_exhaustive:
            ys_a_ex.append(lin2db(acc_a["ex"] / max(1, cnt_a["ex"])) )

        if include_large:
            ys_b_opt.append(lin2db(acc_b["opt"] / max(1, cnt_b["opt"])) )
            ys_b_zf.append(lin2db(acc_b["zf"] / max(1, cnt_b["zf"])) )
            ys_b_rzf.append(lin2db(acc_b["rzf"] / max(1, cnt_b["rzf"])) )
            if include_exhaustive:
                ys_b_ex.append(lin2db(acc_b["ex"] / max(1, cnt_b["ex"])) )

    panel_a = (np.array(ys_a_opt), np.array(ys_a_zf), np.array(ys_a_rzf), np.array(ys_a_ex) if include_exhaustive else None)
    panel_b = None
    if include_large:
        panel_b = (np.array(ys_b_opt), np.array(ys_b_zf), np.array(ys_b_rzf), np.array(ys_b_ex) if include_exhaustive else None)
    return panel_a, panel_b


# ================================ Plotting ================================

def plot_panel(snr_grid_db, panel, cfg: SystemConfig, samples: int, title: str, include_exhaustive: bool):
    y_opt, y_zf, y_rzf, y_ex = panel

    TITLE_FS  = 18
    LABEL_FS  = 16
    TICK_FS   = 14
    LEGEND_FS = 13

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(snr_grid_db, y_opt, "o-", label="Optimal (teacher, CPU)")
    ax.plot(snr_grid_db, y_rzf, "s-", label="RZF (equal power)")
    ax.plot(snr_grid_db, y_zf,  "^-", label="ZF (equal power)")
    if include_exhaustive and y_ex is not None:
        ax.plot(snr_grid_db, y_ex, "d-", label=f"Exhaustive (U≤{min(cfg.U_max,cfg.K,cfg.M)})")

    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(r"Normalized transmit power $10\log_{10}(P_{\max}/\sigma^2)$ (dB)", fontsize=LABEL_FS)
    ax.set_ylabel("Balanced SINR (dB)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)

    ax.legend(loc="upper left", fontsize=LEGEND_FS, frameon=True)
    fig.tight_layout()
    return fig


# ================================ Saving CSV ================================

def save_curves_csv(csv_path, snr_grid_db, panel, include_exhaustive: bool):
    y_opt, y_zf, y_rzf, y_ex = panel
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["snr_dB", "opt_db", "zf_db", "rzf_db"]
        if include_exhaustive:
            header.append("ex_db")
        w.writerow(header)
        for i, snr_db in enumerate(snr_grid_db):
            row = [snr_db, float(y_opt[i]), float(y_zf[i]), float(y_rzf[i])]
            if include_exhaustive:
                row.append(float(y_ex[i]))
            w.writerow(row)
    assert_saved(csv_path, kind="csv")


# ================================ CLI main ================================

def main():
    print("[DEBUG] running:", abspath_here(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200,
                        help="Monte Carlo samples per SNR point")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: cfg.seed)")
    parser.add_argument("--mode", type=str, default="small",
                        choices=["small", "large", "both"],
                        help="Which figure(s) to generate")
    parser.add_argument("--include-exhaustive", action="store_true",
                        help="Include exhaustive curve (CPU, slow)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures + CSV under --outdir")
    parser.add_argument("--outdir", type=str, default="figs",
                        help="Output directory for PNG and CSV (default: figs/)")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display figures")
    parser.add_argument("--snr-max", type=int, default=30,
                        help="Maximum SNR in dB (grid is 0:3:snr-max)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose progress printing")

    args = parser.parse_args()

    cfg = SystemConfig()
    snr_grid_db = list(range(0, args.snr_max + 1, 3))

    print(f"[noise] sigma² = {cfg.sigma2:.3e} W")
    print("[DEBUG] outdir:", abspath_here(args.outdir))
    print("SNR grid (dB):", snr_grid_db)
    print("mode:", args.mode, "| samples:", args.samples)
    print("include_exhaustive:", args.include_exhaustive)

    include_large = (args.mode in ["large", "both"])

    panel_a, panel_b = run_case(
        cfg=cfg,
        samples=args.samples,
        snr_grid_db=snr_grid_db,
        include_large=include_large,
        seed=args.seed,
        verbose=args.verbose,
        include_exhaustive=args.include_exhaustive,
    )

    fig_a = None
    fig_b = None

    if args.mode in ["small", "both"]:
        title_a = f"(a) Small-scale fading only | K={cfg.K}, M={cfg.M}, samples={args.samples}"
        fig_a = plot_panel(snr_grid_db, panel_a, cfg, args.samples, title_a, args.include_exhaustive)

    if args.mode in ["large", "both"]:
        if panel_b is None:
            raise RuntimeError("panel_b is None but mode requires large-scale simulation.")
        title_b = f"(b) Large-scale + small-scale (3GPP UMa) | K={cfg.K}, M={cfg.M}, samples={args.samples}"
        fig_b = plot_panel(snr_grid_db, panel_b, cfg, args.samples, title_b, args.include_exhaustive)

    # ---------------- Save outputs (PNG + CSV) ----------------
    if args.save:
        ensure_dir(args.outdir)

        if fig_a is not None:
            fig_a_path = os.path.join(args.outdir, "fig5_a_gpu_opt.png")
            fig_a.savefig(fig_a_path, dpi=300, bbox_inches="tight")
            print("Saved:", fig_a_path)
            assert_saved(fig_a_path, kind="png")

            csv_a_path = os.path.join(args.outdir, "fig5_a_gpu_opt_curves.csv")
            save_curves_csv(csv_a_path, snr_grid_db, panel_a, args.include_exhaustive)
            print("Saved:", csv_a_path)

        if fig_b is not None:
            fig_b_path = os.path.join(args.outdir, "fig5_b_gpu_opt.png")
            fig_b.savefig(fig_b_path, dpi=300, bbox_inches="tight")
            print("Saved:", fig_b_path)
            assert_saved(fig_b_path, kind="png")

            csv_b_path = os.path.join(args.outdir, "fig5_b_gpu_opt_curves.csv")
            save_curves_csv(csv_b_path, snr_grid_db, panel_b, args.include_exhaustive)
            print("Saved:", csv_b_path)

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
