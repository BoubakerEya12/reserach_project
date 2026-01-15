# scripts/p1_vs_M_UMa.py
# ---------------------------------------------------------------------
# P1 (Balanced SINR) vs M under 3GPP TR 38.901 UMa large-scale model.
# K is fixed; we vary M.
# Methods: Optimal (teacher), ZF eq-power, RZF eq-power, Exhaustive eq-power.
# Uses the common P1/P2/P3 helpers from scripts.plot_fig5.
# ---------------------------------------------------------------------

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sim.channels import (
    gen_small_scale_iid_rayleigh,
    apply_large_scale,
    apply_csit_model,
    noise_power_per_rb,
)
from sim.channels_3gpp import gen_large_scale_3gpp_uma

# Common P1 helpers (teacher + ZF/RZF/Exhaustive definitions)
from scripts.plot_fig5 import (
    lin2db,
    db2lin,
    compute_methods_large_plus_small,
)


# ===================== Monte Carlo for one M =====================

def mc_for_one_M(M, K, N, sigma2, P_rb, samples, rng,
                 U_max, fc_GHz, h_bs, h_ut, cell_radius,
                 eta, verbose=False):
    """
    For a given (M,K), average balanced SINR over 'samples' channel draws
    using the same methods as in plot_fig5.py (teacher + ZF/RZF/Exhaustive).
    """
    if U_max is None:
        U_max = min(M, K)

    acc_opt = 0.0
    acc_zf  = 0.0
    acc_rzf = 0.0
    acc_ex  = 0.0

    for s in range(samples):
        if verbose and s % max(1, samples // 5) == 0:
            print(f"  [M={M}] sample {s}/{samples}")

        # --- Large-scale (3GPP UMa) ---
        beta, is_los, d2d = gen_large_scale_3gpp_uma(
            K=K, fc_GHz=fc_GHz, h_bs=h_bs, h_ut=h_ut,
            los_mode="prob", cell_radius_m=cell_radius, rng=rng
        )  # beta: [K]

        # --- Small-scale ---
        h_ss = gen_small_scale_iid_rayleigh(M, K, N, rng)  # [K,N,M]

        # --- Effective channel (large-scale) ---
        h_eff = apply_large_scale(h_ss, beta)              # [K,N,M]

        # --- BS-side CSIT (optionally imperfect) ---
        h_bs = apply_csit_model(h_eff, eta, rng) if eta < 1.0 else h_eff

        # Single-RB evaluation (RB 0)
        HkM_eff = h_bs[:, 0, :]  # [K,M]

        # Reuse the common evaluator (teacher + ZF + RZF + Exhaustive)
        opt_lin, zf_lin, rzf_lin, ex_lin, _ = compute_methods_large_plus_small(
            HkM_eff, sigma2, P_rb, U_max, I_out=None, with_wmmse=False
        )

        acc_opt += opt_lin
        acc_zf  += zf_lin
        acc_rzf += rzf_lin
        acc_ex  += ex_lin

    return acc_opt/samples, acc_zf/samples, acc_rzf/samples, acc_ex/samples


# ============================== CLI + main ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4, help="Number of users (fixed)")
    ap.add_argument("--M-list", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28, 32],
                    help="List of BS antennas to evaluate, e.g. --M-list 4 8 12 16")
    ap.add_argument("--N", type=int, default=1, help="Number of RBs in small-scale draw (we use RB 0).")
    ap.add_argument("--samples", type=int, default=2000, help="Monte Carlo samples per point.")
    ap.add_argument("--snr-db", type=float, default=10.0,
                    help="Target normalized SNR = 10*log10(P_max / sigma^2).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--N0", type=float, default=1e-20, help="Noise PSD (W/Hz).")
    ap.add_argument("--BRB", type=float, default=180e3, help="RB bandwidth (Hz).")

    ap.add_argument("--fc-GHz", type=float, default=3.5, help="Carrier frequency (GHz).")
    ap.add_argument("--hbs", type=float, default=25.0, help="BS height (m).")
    ap.add_argument("--hut", type=float, default=1.5, help="UE height (m).")
    ap.add_argument("--cell", type=float, default=250.0, help="Cell radius (m).")
    ap.add_argument("--eta", type=float, default=1.0, help="CSIT quality in [0,1]. 1.0=no error.")

    ap.add_argument("--save-fig", action="store_true", help="Save the figure as PNG.")
    ap.add_argument("--no-show", action="store_true", help="Do not display the figure.")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Noise power per RB (same as plot_fig5)
    sigma2 = float(noise_power_per_rb(args.N0, args.BRB))
    print(f"[noise] sigma^2 = {sigma2:.3e} W")

    # RB power corresponding to the target SNR
    P_rb = sigma2 * db2lin(args.snr_db)
    print(f"[power] P_rb = {P_rb:.3e} W  (SNR = {args.snr_db:.1f} dB)")

    Ms = sorted(args.M_list)
    K = args.K
    N = args.N

    y_opt_lin = []
    y_zf_lin  = []
    y_rzf_lin = []
    y_ex_lin  = []

    for M in Ms:
        if args.verbose:
            print(f"\n=== Evaluating M={M}, K={K} (UMa) ===")
        avg_opt, avg_zf, avg_rzf, avg_ex = mc_for_one_M(
            M=M, K=K, N=N, sigma2=sigma2, P_rb=P_rb,
            samples=args.samples, rng=rng, U_max=None,
            fc_GHz=args.fc_GHz, h_bs=args.hbs, h_ut=args.hut,
            cell_radius=args.cell, eta=args.eta, verbose=args.verbose
        )
        y_opt_lin.append(avg_opt)
        y_zf_lin.append(avg_zf)
        y_rzf_lin.append(avg_rzf)
        y_ex_lin.append(avg_ex)

    # Convert to dB
    y_opt_db = lin2db(np.array(y_opt_lin))
    y_zf_db  = lin2db(np.array(y_zf_lin))
    y_rzf_db = lin2db(np.array(y_rzf_lin))
    y_ex_db  = lin2db(np.array(y_ex_lin))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ms, y_opt_db, "o-", label="Optimal (max–min)")
    ax.plot(Ms, y_rzf_db, "s-", label="RZF (equal power)")
    ax.plot(Ms, y_zf_db,  "^-", label="ZF (equal power)")
    ax.plot(Ms, y_ex_db,  "d-", label="Exhaustive equal-power")

    ax.set_xlabel(r"Number of BS antennas $M$")
    ax.set_ylabel("Balanced SINR (dB)")
    ax.set_title(f"P1: Balanced SINR vs M  (K={K}, SNR={args.snr_db:.1f} dB, eta={args.eta})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # CSV
    os.makedirs("excel", exist_ok=True)
    csv_path = os.path.join("excel", f"p1_vs_M_UMa_K{K}_snr{int(args.snr_db)}dB.csv")
    header = "M,Optimal_dB,RZF_dB,ZF_dB,Exhaustive_dB"
    data = np.column_stack([Ms, y_opt_db, y_rzf_db, y_zf_db, y_ex_db])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"[save] CSV data written to {csv_path}")

    # Figure
    if args.save_fig:
        os.makedirs("figs", exist_ok=True)
        fig_path = os.path.join("figs", f"p1_vs_M_UMa_K{K}_snr{int(args.snr_db)}dB.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"[save] Figure saved to {fig_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
