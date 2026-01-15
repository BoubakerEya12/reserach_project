# scripts/plot_fig5.py
# ---------------------------------------------------------------------
# P1: Balanced-SINR maximization (teacher) + Xia-style Fig. 5
#
# NEW in this version:
#  - --mode {small, large, both} to generate figures separately
#  - Saves TWO separate CSV files (one per figure) in the same folder as figures
#  - Larger fonts for title/axes/ticks/legend
#  - Legend location: small-scale -> upper-left (doesn't hide curves)
#  - DEBUG: prints the exact script path being executed + verifies that files exist


#python -m scripts.plot_fig5 --mode small --samples 200 --save --verbose (Small uniquement)

#python -m scripts.plot_fig5 --mode large --samples 200 --save --verbose (Large uniquement)

#python -m scripts.plot_fig5 --mode both --samples 200 --save --verbose  (Les deux)
# ---------------------------------------------------------------------

import os
import csv
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from config import SystemConfig
from scripts.common import (
    make_rng,
    draw_iid_small_scale,   # small-scale Rayleigh H_s: [K, M]
    effective_channel,      # H_eff = sqrt(beta_u) * H_s
)
from sim.channels_3gpp import gen_large_scale_3gpp_uma
from sim.exhaustive_teacher import best_group_by_balanced_sinr


# =============================== DEBUG ===============================

def abspath_here(path: str) -> str:
    return os.path.abspath(path)

def ensure_dir(d: str):
    if d is None or d == "":
        return
    os.makedirs(d, exist_ok=True)

def assert_saved(path: str, kind: str = "file"):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else -1
    print(f"[SAVE-CHECK] {kind}: {abspath_here(path)} | exists={exists} | size={size} bytes")
    if not exists or size <= 0:
        raise RuntimeError(f"[SAVE-CHECK] Failed to save {kind}: {abspath_here(path)}")


# =============================== Helpers ===============================

def lin2db(x):
    x = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(x, 1e-12))

def db2lin(db):
    return 10.0 ** (db / 10.0)

def clip_sinr(x_lin, max_db=60.0):
    """Clip linear SINR to at most max_db (to stabilize averages)."""
    return min(float(x_lin), 10.0 ** (max_db / 10.0))

def sinr_for_selection(HkM, sel, W_full, sigma2):
    """
    SINRs for selected users 'sel'.
    HkM : [K,M], W_full : [M,K] with unused columns = 0.
    Returns sinr_u:[K].
    """
    K, M = HkM.shape
    HW = HkM @ W_full      # [K,K]
    sinr_u = np.zeros(K, dtype=np.float64)
    for u in sel:
        sig = np.abs(HW[u, u]) ** 2
        interf = np.sum(np.abs(HW[u, :]) ** 2) - sig
        sinr_u[u] = sig / max(interf + sigma2, 1e-30)
    return sinr_u

def pick_subset_topU_by_norm(HkM, U_max):
    norms = np.linalg.norm(HkM, axis=1)
    U = min(U_max, HkM.shape[0])
    sel = np.argsort(-norms)[:U]
    return np.sort(sel)

def rzf_directions(H_sel, sigma2, P_rb):
    """
    RZF directions: V = H (H^H H + α I)^(-1), columns normalized.
    H_sel: [U,M]; return V: [M,U] with unit-norm columns.
    """
    U, M = H_sel.shape
    H = H_sel.T                      # [M,U]
    Gram = H.conj().T @ H            # [U,U]
    alpha = (U * sigma2 / max(P_rb, 1e-12))
    A = Gram + alpha * np.eye(U, dtype=Gram.dtype)
    V = H @ np.linalg.pinv(A)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    return V.astype(np.complex64)

def build_equal_power_beams(H_sel, sigma2, P_rb, mode="rzf"):
    """
    Equal-power beams for selected users (subset H_sel [U,M]).
    Returns (V, p, W) with:
       V:[M,U] unit-norm directions
       p:[U]   per-user powers (equal)
       W:[M,U] actual precoder V*sqrt(p)^T
    """
    U, M = H_sel.shape
    if mode == "zf":
        H = H_sel.T
        Gram = H.conj().T @ H
        V = H @ np.linalg.pinv(Gram)
        V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    else:  # rzf
        V = rzf_directions(H_sel, sigma2, P_rb)

    p = np.full(U, P_rb / max(U, 1), dtype=np.float64)
    W = V * np.sqrt(p.reshape(1, -1))
    return V.astype(np.complex64), p.astype(np.float64), W.astype(np.complex64)

def zf_equal_power_value(HkM, sigma2, P_rb, U_max):
    sel = pick_subset_topU_by_norm(HkM, U_max)
    H_sel = HkM[sel, :]
    V, p, W = build_equal_power_beams(H_sel, sigma2, P_rb, mode="zf")
    K, M = HkM.shape
    W_full = np.zeros((M, K), dtype=np.complex64)
    for j, u in enumerate(sel):
        W_full[:, u] = W[:, j]
    sinr_u = sinr_for_selection(HkM, sel, W_full, sigma2)
    min_lin = float(np.min(sinr_u[sel])) if sel.size > 0 else 0.0
    return max(min_lin, 1e-30)

def rzf_equal_power_value(HkM, sigma2, P_rb, U_max):
    sel = pick_subset_topU_by_norm(HkM, U_max)
    H_sel = HkM[sel, :]
    V, p, W = build_equal_power_beams(H_sel, sigma2, P_rb, mode="rzf")
    K, M = HkM.shape
    W_full = np.zeros((M, K), dtype=np.complex64)
    for j, u in enumerate(sel):
        W_full[:, u] = W[:, j]
    sinr_u = sinr_for_selection(HkM, sel, W_full, sigma2)
    min_lin = float(np.min(sinr_u[sel])) if sel.size > 0 else 0.0
    return max(min_lin, 1e-30)

def exhaustive_equal_power_value(HkM, sigma2, P_rb, U_max, mode="rzf"):
    """
    Exhaustive equal-power upper bound among all subsets up to U_max.
    """
    K, M = HkM.shape
    best = -np.inf
    users = list(range(K))
    for U in range(1, min(U_max, K) + 1):
        for comb in itertools.combinations(users, U):
            sel = np.array(comb, dtype=int)
            H_sel = HkM[sel, :]
            V, p, W = build_equal_power_beams(H_sel, sigma2, P_rb, mode=mode)
            W_full = np.zeros((M, K), dtype=np.complex64)
            for j, u in enumerate(sel):
                W_full[:, u] = W[:, j]
            sinr_u = sinr_for_selection(HkM, sel, W_full, sigma2)
            min_lin = float(np.min(sinr_u[sel]))
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
             verbose: bool = False):
    """
    Run Monte-Carlo for all SNR points.
    Returns:
      panel_a = (opt, zf, rzf, ex)  [in dB arrays]
      panel_b = (opt, zf, rzf, ex)  [in dB arrays] or None
    """
    rng = make_rng(seed if seed is not None else cfg.seed)

    K = cfg.K
    M = cfg.M
    sigma2 = cfg.sigma2
    U_max = min(cfg.U_max, K, M)

    ys_a_opt, ys_a_zf, ys_a_rzf, ys_a_ex = [], [], [], []
    ys_b_opt, ys_b_zf, ys_b_rzf, ys_b_ex = [], [], [], []

    for snr_db in snr_grid_db:
        P_rb = sigma2 * db2lin(snr_db)

        acc_a = {"opt": 0.0, "zf": 0.0, "rzf": 0.0, "ex": 0.0}
        cnt_a = {"opt": 0,    "zf": 0,    "rzf": 0,    "ex": 0}

        acc_b = {"opt": 0.0, "zf": 0.0, "rzf": 0.0, "ex": 0.0}
        cnt_b = {"opt": 0,    "zf": 0,    "rzf": 0,    "ex": 0}

        if verbose:
            print(f"[P1] SNR={snr_db} dB, sample 0/{samples}")

        for s in range(samples):
            if verbose and (s % max(1, samples // 5) == 0) and s > 0:
                print(f"[P1] SNR={snr_db} dB, sample {s}/{samples}")

            # ---------- Small-scale only ----------
            Hs = draw_iid_small_scale(rng, K, M)  # [K,M]
            HkM = Hs

            sel_o, W_o, p_o, sinr_o, score_lin0 = best_group_by_balanced_sinr(
                Hn=HkM.T,
                sigma2=sigma2,
                P_rb=P_rb,
                U_max=U_max,
            )
            if sel_o is not None and sel_o.size > 0:
                opt_lin = float(np.min(sinr_o[sel_o]))
            else:
                opt_lin = float(score_lin0)

            zf_lin  = zf_equal_power_value(HkM, sigma2, P_rb, U_max)
            rzf_lin = rzf_equal_power_value(HkM, sigma2, P_rb, U_max)
            ex_lin  = exhaustive_equal_power_value(HkM, sigma2, P_rb, U_max, mode="rzf")

            opt_lin = clip_sinr(opt_lin)
            zf_lin  = clip_sinr(zf_lin)
            rzf_lin = clip_sinr(rzf_lin)
            ex_lin  = clip_sinr(ex_lin)

            acc_a["opt"] += opt_lin; cnt_a["opt"] += 1
            acc_a["zf"]  += zf_lin;  cnt_a["zf"]  += 1
            acc_a["rzf"] += rzf_lin; cnt_a["rzf"] += 1
            acc_a["ex"]  += ex_lin;  cnt_a["ex"]  += 1

            # ---------- Large-scale (3GPP UMa) + small-scale ----------
            if include_large:
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
                HkM_eff = H_eff

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

                zf_lin_b  = zf_equal_power_value(HkM_eff, sigma2, P_rb, U_max)
                rzf_lin_b = rzf_equal_power_value(HkM_eff, sigma2, P_rb, U_max)
                ex_lin_b  = exhaustive_equal_power_value(HkM_eff, sigma2, P_rb, U_max, mode="rzf")

                opt_lin_b = clip_sinr(opt_lin_b)
                zf_lin_b  = clip_sinr(zf_lin_b)
                rzf_lin_b = clip_sinr(rzf_lin_b)
                ex_lin_b  = clip_sinr(ex_lin_b)

                acc_b["opt"] += opt_lin_b; cnt_b["opt"] += 1
                acc_b["zf"]  += zf_lin_b;  cnt_b["zf"]  += 1
                acc_b["rzf"] += rzf_lin_b; cnt_b["rzf"] += 1
                acc_b["ex"]  += ex_lin_b;  cnt_b["ex"]  += 1

        ys_a_opt.append(lin2db(acc_a["opt"] / max(1, cnt_a["opt"])))
        ys_a_zf .append(lin2db(acc_a["zf"]  / max(1, cnt_a["zf"])))
        ys_a_rzf.append(lin2db(acc_a["rzf"] / max(1, cnt_a["rzf"])))
        ys_a_ex .append(lin2db(acc_a["ex"]  / max(1, cnt_a["ex"])))

        if include_large:
            ys_b_opt.append(lin2db(acc_b["opt"] / max(1, cnt_b["opt"])))
            ys_b_zf .append(lin2db(acc_b["zf"]  / max(1, cnt_b["zf"])))
            ys_b_rzf.append(lin2db(acc_b["rzf"] / max(1, cnt_b["rzf"])))
            ys_b_ex .append(lin2db(acc_b["ex"]  / max(1, cnt_b["ex"])))

    panel_a = (np.array(ys_a_opt), np.array(ys_a_zf), np.array(ys_a_rzf), np.array(ys_a_ex))
    panel_b = None
    if include_large:
        panel_b = (np.array(ys_b_opt), np.array(ys_b_zf), np.array(ys_b_rzf), np.array(ys_b_ex))
    return panel_a, panel_b


# ================================ Plotting ================================

def plot_panel_a(snr_grid_db, panel_a, cfg: SystemConfig, samples: int):
    y_opt, y_zf, y_rzf, y_ex = panel_a

    TITLE_FS  = 18
    LABEL_FS  = 16
    TICK_FS   = 14
    LEGEND_FS = 13

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(snr_grid_db, y_opt, "o-", label="Optimal (max-min)")
    ax.plot(snr_grid_db, y_rzf, "s-", label="RZF (equal power)")
    ax.plot(snr_grid_db, y_zf,  "^-", label="ZF (equal power)")
    ax.plot(snr_grid_db, y_ex,  "d-", label=f"Exhaustive (U≤{min(cfg.U_max,cfg.K,cfg.M)})")

    ax.set_title(f"(a) Small-scale fading only  |  K={cfg.K}, M={cfg.M}, samples={samples}",
                 fontsize=TITLE_FS)
    ax.set_xlabel(r"Normalized transmit power $10\log_{10}(P_{\max}/\sigma^2)$ (dB)",
                  fontsize=LABEL_FS)
    ax.set_ylabel("Balanced SINR (dB)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)

    ax.legend(loc="upper left", fontsize=LEGEND_FS, frameon=True)
    fig.tight_layout()
    return fig

def plot_panel_b(snr_grid_db, panel_b, cfg: SystemConfig, samples: int):
    y_opt, y_zf, y_rzf, y_ex = panel_b

    TITLE_FS  = 18
    LABEL_FS  = 16
    TICK_FS   = 14
    LEGEND_FS = 13

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(snr_grid_db, y_opt, "o-", label="Optimal (max-min)")
    ax.plot(snr_grid_db, y_rzf, "s-", label="RZF (equal power)")
    ax.plot(snr_grid_db, y_zf,  "^-", label="ZF (equal power)")
    ax.plot(snr_grid_db, y_ex,  "d-", label=f"Exhaustive (U≤{min(cfg.U_max,cfg.K,cfg.M)})")

    ax.set_title(f"(b) Large-scale + small-scale (3GPP UMa)  |  K={cfg.K}, M={cfg.M}, samples={samples}",
                 fontsize=TITLE_FS)
    ax.set_xlabel(r"Normalized transmit power $10\log_{10}(P_{\max}/\sigma^2)$ (dB)",
                  fontsize=LABEL_FS)
    ax.set_ylabel("Balanced SINR (dB)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)

    ax.legend(loc="lower right", fontsize=LEGEND_FS, frameon=True)
    fig.tight_layout()
    return fig


# ================================ Saving CSV ================================

def save_curves_csv(csv_path, snr_grid_db, panel):
    """
    Save one panel curves to CSV.
    Columns: snr_dB, opt_db, zf_db, rzf_db, ex_db  (all in dB)
    """
    y_opt, y_zf, y_rzf, y_ex = panel
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_dB", "opt_db", "zf_db", "rzf_db", "ex_db"])
        for i, snr_db in enumerate(snr_grid_db):
            w.writerow([snr_db, float(y_opt[i]), float(y_zf[i]), float(y_rzf[i]), float(y_ex[i])])

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

    include_large = (args.mode in ["large", "both"])

    panel_a, panel_b = run_case(
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
        fig_a = plot_panel_a(snr_grid_db, panel_a, cfg, args.samples)

    if args.mode in ["large", "both"]:
        if panel_b is None:
            raise RuntimeError("panel_b is None but mode requires large-scale simulation.")
        fig_b = plot_panel_b(snr_grid_db, panel_b, cfg, args.samples)

    # ---------------- Save outputs (PNG + CSV) ----------------
    if args.save:
        ensure_dir(args.outdir)

        if fig_a is not None:
            fig_a_path = os.path.join(args.outdir, "fig5_a.png")
            fig_a.savefig(fig_a_path, dpi=300, bbox_inches="tight")
            print("Saved:", fig_a_path)
            assert_saved(fig_a_path, kind="png")

            csv_a_path = os.path.join(args.outdir, "fig5_a_curves.csv")
            save_curves_csv(csv_a_path, snr_grid_db, panel_a)
            print("Saved:", csv_a_path)

        if fig_b is not None:
            fig_b_path = os.path.join(args.outdir, "fig5_b.png")
            fig_b.savefig(fig_b_path, dpi=300, bbox_inches="tight")
            print("Saved:", fig_b_path)
            assert_saved(fig_b_path, kind="png")

            csv_b_path = os.path.join(args.outdir, "fig5_b_curves.csv")
            save_curves_csv(csv_b_path, snr_grid_db, panel_b)
            print("Saved:", csv_b_path)

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
