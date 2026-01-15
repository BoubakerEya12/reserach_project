# ---------------------------------------------------------------------
# P1 (Balanced SINR) vs K under 3GPP TR 38.901 UMa large-scale model.
#
# Methods:
#   - Optimal (teacher P1)
#   - ZF (equal power)
#   - RZF (equal power)
#   - Exhaustive equal-power
#
# Uses unified SystemConfig for noise, pathloss and CSIT.
# ---------------------------------------------------------------------

import os
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sim.channels import (
    gen_small_scale_iid_rayleigh,  # h: [K, N, M] ~ CN(0, I)
    apply_large_scale,             # h_eff = sqrt(beta)*h
    apply_csit_model,              # \hat{h} = eta*h_eff + sqrt(1-eta^2)*e
)
from sim.channels_3gpp import gen_large_scale_3gpp_uma
from sim.exhaustive_teacher import best_group_by_balanced_sinr
from sim.config import SystemConfig


# ============================ Small helpers ============================

def lin2db(x):
    x = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(x, 1e-12))

def db2lin(db):
    return 10.0 ** (db / 10.0)


def rzf_directions(H_sel, sigma2, P_rb):
    """RZF directions for selected users H_sel [U, M]."""
    U, M = H_sel.shape
    H = H_sel.T  # [M,U]
    Gram = H.conj().T @ H           # [U,U]
    alpha = (U * sigma2 / max(P_rb, 1e-12))
    A = Gram + alpha * np.eye(U, dtype=Gram.dtype)
    V = H @ np.linalg.pinv(A)       # [M,U]
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    return V.astype(np.complex64)


def build_equal_power_beams(H_sel, sigma2, P_rb, mode="rzf"):
    """
    Build equal-power beams (ZF or RZF) for users in H_sel [U, M].
    Returns:
      V : unit-norm directions [M,U]
      p : per-user powers [U]
      W : full precoder [M,U] = V * sqrt(p)^T
    """
    U, M = H_sel.shape
    H = H_sel.T  # [M,U]

    if mode == "zf":
        Gram = H.conj().T @ H  # [U,U]
        V = H @ np.linalg.pinv(Gram)
        V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    else:
        V = rzf_directions(H_sel, sigma2, P_rb)

    p = np.full(U, P_rb / max(U, 1), dtype=np.float64)
    W = V * np.sqrt(p.reshape(1, -1))
    return V.astype(np.complex64), p.astype(np.float64), W.astype(np.complex64)


def sinr_for_selection(HkM, sel, W_full, sigma2):
    """
    Compute SINR per user for a given selection.
    HkM   : [K,M], channels (K users)
    sel   : chosen users (indices)
    W_full: [M,K], full precoder (columns for all K; zeros for unserved)
    sigma2: noise power
    """
    K, M = HkM.shape
    sinr_u = np.zeros(K, dtype=np.float64)
    if sel.size == 0:
        return sinr_u

    HW = HkM @ W_full  # [K,K]
    for u in sel:
        sig = np.abs(HW[u, u])**2
        interf = np.sum(np.abs(HW[u, sel])**2) - np.abs(HW[u, u])**2
        denom = interf + sigma2
        sinr_u[u] = sig / max(denom, 1e-30)
    return sinr_u


def pick_subset_topU_by_norm(HkM, U_max):
    norms = np.linalg.norm(HkM, axis=1)
    U = min(U_max, HkM.shape[0])
    sel = np.argsort(-norms)[:U]
    return np.sort(sel)


def zf_equal_power_value(HkM, sigma2, P_rb, U_max):
    sel = pick_subset_topU_by_norm(HkM, U_max)
    H_sel = HkM[sel, :]
    _, _, W = build_equal_power_beams(H_sel, sigma2, P_rb, mode="zf")

    K, M = HkM.shape
    W_full = np.zeros((M, K), dtype=np.complex64)
    for j, u in enumerate(sel):
        W_full[:, u] = W[:, j]

    sinr_u = sinr_for_selection(HkM, sel, W_full, sigma2)
    min_lin = float(np.min(sinr_u[sel]) if sel.size > 0 else 0.0)
    return max(min_lin, 1e-30)


def rzf_equal_power_value(HkM, sigma2, P_rb, U_max):
    sel = pick_subset_topU_by_norm(HkM, U_max)
    H_sel = HkM[sel, :]
    _, _, W = build_equal_power_beams(H_sel, sigma2, P_rb, mode="rzf")

    K, M = HkM.shape
    W_full = np.zeros((M, K), dtype=np.complex64)
    for j, u in enumerate(sel):
        W_full[:, u] = W[:, j]

    sinr_u = sinr_for_selection(HkM, sel, W_full, sigma2)
    min_lin = float(np.min(sinr_u[sel]) if sel.size > 0 else 0.0)
    return max(min_lin, 1e-30)


def exhaustive_equal_power_value(HkM, sigma2, P_rb, U_max, mode="rzf"):
    K, M = HkM.shape
    users = list(range(K))
    best_min = 0.0

    for U in range(1, min(U_max, K) + 1):
        for comb in itertools.combinations(users, U):
            sel = np.array(comb, dtype=int)
            H_sel = HkM[sel, :]
            _, _, W = build_equal_power_beams(H_sel, sigma2, P_rb, mode=mode)

            W_full = np.zeros((M, K), dtype=np.complex64)
            for j, u in enumerate(sel):
                W_full[:, u] = W[:, j]

            sinr_u = sinr_for_selection(HkM, sel, W_full, sigma2)
            min_lin = float(np.min(sinr_u[sel]))
            if min_lin > best_min:
                best_min = min_lin

    return max(best_min, 1e-30)


# ===================== Monte Carlo for one K =====================

def mc_for_one_K(K, M, N, sigma2, P_rb, samples, rng,
                 U_max, fc_GHz, h_bs, h_ut, cell_radius, eta, verbose=False):
    if U_max is None:
        U_max = min(M, K)

    acc_opt = 0.0
    acc_zf  = 0.0
    acc_rzf = 0.0
    acc_ex  = 0.0

    for s in range(samples):
        if verbose and s % max(1, samples // 5) == 0:
            print(f"  [K={K}] sample {s}/{samples}")

        # --- Large-scale (UMa) ---
        beta, is_los, d2d = gen_large_scale_3gpp_uma(
            K=K, fc_GHz=fc_GHz, h_bs=h_bs, h_ut=h_ut,
            los_mode="prob", cell_radius_m=cell_radius, rng=rng
        )

        # --- Small-scale ---
        h_ss = gen_small_scale_iid_rayleigh(M, K, N, rng)  # [K,N,M]

        # --- Effective channels + optional CSIT error ---
        h_eff = apply_large_scale(h_ss, beta)              # [K,N,M]
        h_bs  = apply_csit_model(h_eff, eta, rng) if eta < 1.0 else h_eff

        # Use RB 0 for single-RB evaluation
        HkM = h_bs[:, 0, :]  # [K,M]

        # --- Optimal teacher ---
        sel_opt, W_opt, p_opt, sinr_opt, score_lin = best_group_by_balanced_sinr(
            Hn=HkM.T, sigma2=sigma2, P_rb=P_rb, U_max=U_max
        )
        opt_lin = float(max(score_lin, 1e-30))

        # --- Baselines ---
        zf_lin  = zf_equal_power_value(HkM, sigma2, P_rb, U_max)
        rzf_lin = rzf_equal_power_value(HkM, sigma2, P_rb, U_max)
        ex_lin  = exhaustive_equal_power_value(HkM, sigma2, P_rb, U_max, mode="rzf")

        acc_opt += opt_lin
        acc_zf  += zf_lin
        acc_rzf += rzf_lin
        acc_ex  += ex_lin

    return acc_opt/samples, acc_zf/samples, acc_rzf/samples, acc_ex/samples


# ============================== CLI + main ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=None,
                    help="Number of BS antennas (fixed). If None, use cfg.M.")
    ap.add_argument("--K-list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8],
                    help="List of user counts to evaluate, e.g. --K-list 2 3 4 5 6 8")
    ap.add_argument("--N", type=int, default=None,
                    help="Number of RBs in small-scale draw (if None, use cfg.N_RB).")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--snr-db", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--eta", type=float, default=None)

    ap.add_argument("--save-fig", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    cfg   = SystemConfig()
    M     = args.M if args.M is not None else cfg.M
    N     = args.N if args.N is not None else cfg.N_RB
    eta   = args.eta if args.eta is not None else cfg.eta
    seed  = args.seed if args.seed is not None else cfg.seed

    rng    = np.random.default_rng(seed)
    sigma2 = cfg.sigma2
    P_rb   = sigma2 * db2lin(args.snr_db)

    print(f"[noise] sigma^2 = {sigma2:.3e} W")
    print(f"[power] P_rb = {P_rb:.3e} W  (SNR = {args.snr_db:.1f} dB)")

    Ks = sorted(args.K_list)

    y_opt_lin = []
    y_zf_lin  = []
    y_rzf_lin = []
    y_ex_lin  = []

    for K in Ks:
        if args.verbose:
            print(f"\n=== Evaluating K={K}, M={M} (UMa) ===")
        avg_opt, avg_zf, avg_rzf, avg_ex = mc_for_one_K(
            K=K, M=M, N=N, sigma2=sigma2, P_rb=P_rb,
            samples=args.samples, rng=rng, U_max=None,
            fc_GHz=cfg.fc_GHz, h_bs=cfg.h_bs, h_ut=cfg.h_ut,
            cell_radius=cfg.cell_radius_m, eta=eta,
            verbose=args.verbose
        )
        y_opt_lin.append(avg_opt)
        y_zf_lin.append(avg_zf)
        y_rzf_lin.append(avg_rzf)
        y_ex_lin.append(avg_ex)

    y_opt_db = lin2db(np.array(y_opt_lin))
    y_zf_db  = lin2db(np.array(y_zf_lin))
    y_rzf_db = lin2db(np.array(y_rzf_lin))
    y_ex_db  = lin2db(np.array(y_ex_lin))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ks, y_opt_db, "o-", label="Optimal (max–min)")
    ax.plot(Ks, y_rzf_db, "s-", label="RZF (equal power)")
    ax.plot(Ks, y_zf_db,  "^-", label="ZF (equal power)")
    ax.plot(Ks, y_ex_db,  "d-", label="Exhaustive equal-power")

    ax.set_xlabel(r"Number of users $K$")
    ax.set_ylabel("Balanced SINR (dB)")
    ax.set_title(f"P1: Balanced SINR vs K  (M={M}, SNR={args.snr_db:.1f} dB, eta={eta})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    os.makedirs("excel", exist_ok=True)
    csv_path = os.path.join("excel", f"p1_vs_K_UMa_M{M}_snr{int(args.snr_db)}dB.csv")
    header = "K,Optimal_dB,RZF_dB,ZF_dB,Exhaustive_dB"
    data = np.column_stack([Ks, y_opt_db, y_rzf_db, y_zf_db, y_ex_db])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"[save] CSV data written to {csv_path}")

    if args.save_fig:
        os.makedirs("figs", exist_ok=True)
        fig_path = os.path.join("figs", f"p1_vs_K_UMa_M{M}_snr{int(args.snr_db)}dB.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"[save] Figure saved to {fig_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
