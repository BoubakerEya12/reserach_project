# scripts/plot_fig5_opt.py
# ---------------------------------------------------------------------
# P1: Balanced-SINR maximization (teacher) -- OPTIMAL ONLY
#
# Generates Xia-style Fig. 5 with ONLY the Optimal (max-min) curve.
#
# Examples:
#   python -m scripts.plot_fig5_opt --mode small --samples 200 --save --outdir figs/p1_opt_small
#   python -m scripts.plot_fig5_opt --mode large --samples 200 --save --outdir figs/p1_opt_large
#   python -m scripts.plot_fig5_opt --mode both  --samples 200 --save --outdir figs/p1_opt_both
#
# Notes:
#  - Small-scale panel (a): Rayleigh only
#  - Large-scale panel (b): 3GPP UMa large-scale + Rayleigh
#  - Output CSV columns: snr_dB,opt_db
# ---------------------------------------------------------------------

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from config import SystemConfig
from scripts.common import (
    make_rng,
    draw_iid_small_scale,   # H_s: [K, M]
)
from sim.sionna_channel import generate_h_rb_true
from sim.exhaustive_teacher import best_group_by_balanced_sinr


# =============================== utils ===============================

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

def db2lin(db: float) -> float:
    return 10.0 ** (db / 10.0)

def lin2db(x) -> float:
    x = float(x)
    return 10.0 * np.log10(max(x, 1e-12))

def clip_sinr(x_lin: float, max_db: float = 60.0) -> float:
    """Clip linear SINR to at most max_db (stability)."""
    return min(float(x_lin), 10.0 ** (max_db / 10.0))


# ========================== core simulation ==========================

def run_case_opt_only(
    cfg: SystemConfig,
    samples: int,
    snr_grid_db,
    include_large: bool,
    seed: int | None = None,
    verbose: bool = False,
):
    """
    Returns:
      y_opt_a_db : np.ndarray, shape [len(snr_grid_db)]  (panel a)
      y_opt_b_db : np.ndarray or None                    (panel b)
    """
    rng = make_rng(seed if seed is not None else cfg.seed)

    K = cfg.K
    M = cfg.M
    sigma2 = cfg.sigma2
    U_max = min(cfg.U_max, K, M)

    y_opt_a_db = []
    y_opt_b_db = [] if include_large else None

    # Pre-generate 3GPP UMa channels via Sionna (RB-level) if requested.
    # We select one RB index (default 0) to match the single-RB evaluation here.
    HkM_eff_samples = None
    if include_large:
        if seed is not None:
            tf.random.set_seed(seed)
        h_rb_true = generate_h_rb_true(cfg, batch_size=samples)  # [B,K,N_RB,Nt]
        h_rb_true_np = h_rb_true.numpy()
        rb_idx = int(getattr(cfg, "rb_index", 0))
        if rb_idx < 0 or rb_idx >= int(cfg.N_RB):
            rb_idx = 0
        HkM_eff_samples = h_rb_true_np[:, :, rb_idx, :]  # [B,K,M]

    for snr_db in snr_grid_db:
        P_rb = sigma2 * db2lin(snr_db)

        acc_opt_a = 0.0
        cnt_opt_a = 0

        acc_opt_b = 0.0
        cnt_opt_b = 0

        if verbose    :
            print(f"[P1-OPT] SNR={snr_db} dB | samples={samples}")

        for s in range(samples):
            if verbose and samples >= 20 and (s % max(1, samples // 5) == 0) and s > 0:
                print(f"   sample {s}/{samples}")

            # ---------------- Small-scale only (panel a) ----------------
            Hs = draw_iid_small_scale(rng, K, M)  # [K,M]
            HkM = Hs

            sel_o, W_o, p_o, sinr_o, score_lin0 = best_group_by_balanced_sinr(
                Hn=HkM.T,          # expected [M,K]
                sigma2=sigma2,
                P_rb=P_rb,
                U_max=U_max,
            )
            if sel_o is not None and sel_o.size > 0:
                opt_lin = float(np.min(sinr_o[sel_o]))
            else:
                opt_lin = float(score_lin0)

            opt_lin = clip_sinr(opt_lin)
            acc_opt_a += opt_lin
            cnt_opt_a += 1

            # -------- Large-scale + small-scale (panel b, optional) -------
            if include_large:
                HkM_eff = HkM_eff_samples[s]

                sel_b, W_b, p_b, sinr_b, score_lin_b = best_group_by_balanced_sinr(
                    Hn=HkM_eff.T,
                    sigma2=sigma2,
                    P_rb=P_rb,
                    U_max=U_max,
                )
                if sel_b is not None and sel_b.size > 0:
                    opt_lin_b = float(np.min(sinr_b[sel_b]))
                else:
                    opt_lin_b = float(score_lin_b)

                opt_lin_b = clip_sinr(opt_lin_b)
                acc_opt_b += opt_lin_b
                cnt_opt_b += 1

        # Average (linear) then convert to dB
        mean_opt_a = acc_opt_a / max(1, cnt_opt_a)
        y_opt_a_db.append(lin2db(mean_opt_a))

        if include_large:
            mean_opt_b = acc_opt_b / max(1, cnt_opt_b)
            y_opt_b_db.append(lin2db(mean_opt_b))

    y_opt_a_db = np.array(y_opt_a_db, dtype=np.float64)
    if include_large:
        y_opt_b_db = np.array(y_opt_b_db, dtype=np.float64)

    return y_opt_a_db, y_opt_b_db


# ========================== plotting + saving ==========================

def plot_panel_opt(snr_grid_db, y_opt_db, title, cfg: SystemConfig, samples: int, legend_loc="best"):
    TITLE_FS  = 18
    LABEL_FS  = 16
    TICK_FS   = 14
    LEGEND_FS = 13

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(snr_grid_db, y_opt_db, "o-", label="Optimal (max-min)")

    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(r"Normalized transmit power $10\log_{10}(P_{\max}/\sigma^2)$ (dB)", fontsize=LABEL_FS)
    ax.set_ylabel("Balanced SINR (dB)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)
    ax.legend(loc=legend_loc, fontsize=LEGEND_FS, frameon=True)
    fig.tight_layout()
    return fig

def save_csv_opt(csv_path, snr_grid_db, y_opt_db):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_dB", "opt_db"])
        for i, snr_db in enumerate(snr_grid_db):
            w.writerow([float(snr_db), float(y_opt_db[i])])
    assert_saved(csv_path, kind="csv")


# ================================ main ================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200,
                        help="Monte Carlo samples per SNR point")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: cfg.seed)")
    parser.add_argument("--mode", type=str, default="small",
                        choices=["small", "large", "both"],
                        help="Which figure(s) to generate")
    parser.add_argument("--save", action="store_true",
                        help="Save figures + CSV under --outdir")
    parser.add_argument("--outdir", type=str, default="figs",
                        help="Output directory for PNG and CSV")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display figures")
    parser.add_argument("--snr-max", type=int, default=30,
                        help="Maximum SNR in dB (grid is 0:3:snr-max)")
    parser.add_argument("--snr-step", type=int, default=3,
                        help="SNR step in dB (default 3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose progress printing")
    args = parser.parse_args()

    print("[DEBUG] running:", abspath_here(__file__))

    cfg = SystemConfig()
    snr_grid_db = list(range(0, args.snr_max + 1, args.snr_step))
    include_large = (args.mode in ["large", "both"])

    print(f"[noise] sigma² = {cfg.sigma2:.3e} W")
    print("SNR grid (dB):", snr_grid_db)
    print("mode:", args.mode, "| samples:", args.samples)
    print("[DEBUG] outdir:", abspath_here(args.outdir))

    # Run simulation
    y_opt_a_db, y_opt_b_db = run_case_opt_only(
        cfg=cfg,
        samples=args.samples,
        snr_grid_db=snr_grid_db,
        include_large=include_large,
        seed=args.seed,
        verbose=args.verbose,
    )

    fig_a = None
    fig_b = None

    if args.mode in ["small", "both"]:
        title_a = f"(a) Small-scale fading only | K={cfg.K}, M={cfg.M}, samples={args.samples}"
        fig_a = plot_panel_opt(snr_grid_db, y_opt_a_db, title_a, cfg, args.samples, legend_loc="upper left")

    if args.mode in ["large", "both"]:
        if y_opt_b_db is None:
            raise RuntimeError("y_opt_b_db is None but mode requires large-scale simulation.")
        title_b = f"(b) Large-scale + small-scale (3GPP UMa) | K={cfg.K}, M={cfg.M}, samples={args.samples}"
        fig_b = plot_panel_opt(snr_grid_db, y_opt_b_db, title_b, cfg, args.samples, legend_loc="lower right")

    # Save outputs
    if args.save:
        ensure_dir(args.outdir)

        if fig_a is not None:
            fig_a_path = os.path.join(args.outdir, "fig5_a_opt.png")
            fig_a.savefig(fig_a_path, dpi=300, bbox_inches="tight")
            print("Saved:", fig_a_path)
            assert_saved(fig_a_path, kind="png")

            csv_a_path = os.path.join(args.outdir, "fig5_a_opt.csv")
            save_csv_opt(csv_a_path, snr_grid_db, y_opt_a_db)
            print("Saved:", csv_a_path)

        if fig_b is not None:
            fig_b_path = os.path.join(args.outdir, "fig5_b_opt.png")
            fig_b.savefig(fig_b_path, dpi=300, bbox_inches="tight")
            print("Saved:", fig_b_path)
            assert_saved(fig_b_path, kind="png")

            csv_b_path = os.path.join(args.outdir, "fig5_b_opt.csv")
            save_csv_opt(csv_b_path, snr_grid_db, y_opt_b_db)
            print("Saved:", csv_b_path)

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
