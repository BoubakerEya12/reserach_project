# scripts/p1_vs_K_UMa.py
# ---------------------------------------------------------------------
# P1 (Balanced SINR) vs K under 3GPP TR 38.901 UMa large-scale model.
# M is fixed; we vary K.
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

from scripts.plot_fig5 import (
    lin2db,
    db2lin,
    compute_methods_large_plus_small,
)


# ===================== Monte Carlo for one K =====================

def mc_for_one_K(K, M, N, sigma2, P_rb, samples, rng,
                 U_max, fc_GHz, h_bs, h_ut, cell_radius,
                 eta, verbose=False):
    if U_max is None:
        U_max = min(M, K)

    acc_opt = acc_zf = acc_rzf = acc_ex = 0.0

    for s in range(samples):
        if verbose and s % max(1, samples // 5) == 0:
            print(f"  [K={K}] sample {s}/{samples}")

        # Large-scale
        beta, is_los, d2d = gen_large_scale_3gpp_uma(
            K=K, fc_GHz=fc_GHz, h_bs=h_bs, h_ut=h_ut,
            los_mode="prob", cell_radius_m=cell_radius, rng=rng
        )

        # Small-scale
        h_ss = gen_small_scale_iid_rayleigh(M, K, N, rng)  # [K,N,M]

        # Effective channel + CSIT
        h_eff = apply_large_scale(h_ss, beta)              # [K,N,M]
        h_bs  = apply_csit_model(h_eff, eta, rng) if eta < 1.0 else h_eff

        HkM_eff = h_bs[:, 0, :]                            # [K,M]

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
    ap.add_argument("--M", type=int, default=6, help="Number of BS antennas (fixed).")
    ap.add_argument("--K-list", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8],
                    help="List of user counts to evaluate, e.g. --K-list 2 3 4 5 6 8")
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--snr-db", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--N0", type=float, default=1e-20)
    ap.add_argument("--BRB", type=float, default=180e3)

    ap.add_argument("--fc-GHz", type=float, default=3.5, help="Carrier frequency (GHz).")
    ap.add_argument("--hbs", type=float, default=25.0, help="BS height (m).")
    ap.add_argument("--hut", type=float, default=1.5, help="UE height (m).")
    ap.add_argument("--cell", type=float, default=250.0, help="Cell radius (m).")
    ap.add_argument("--eta", type=float, default=1.0, help="CSIT quality in [0,1]. 1.0=no error.")

    ap.add_argument("--save-fig", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    sigma2 = float(noise_power_per_rb(args.N0, args.BRB))
    print(f"[noise] sigma^2 = {sigma2:.3e} W")

    P_rb = sigma2 * db2lin(args.snr_db)
    print(f"[power] P_rb = {P_rb:.3e} W  (SNR = {args.snr_db:.1f} dB)")

    Ks = sorted(args.K_list)
    M  = args.M
    N  = args.N

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
            fc_GHz=args.fc_GHz, h_bs=args.hbs, h_ut=args.hut,
            cell_radius=args.cell, eta=args.eta, verbose=args.verbose
        )
        y_opt_lin.append(avg_opt)
        y_zf_lin.append(avg_zf)
        y_rzf_lin.append(avg_rzf)
        y_ex_lin.append(avg_ex)

    y_opt_db = lin2db(np.array(y_opt_lin))
    y_zf_db  = lin2db(np.array(y_zf_lin))
    y_rzf_db = lin2db(np.array(y_rzf_lin))
    y_ex_db  = lin2db(np.array(y_ex_lin))

    # ===================== Plot styling =====================
    TITLE_FS  = 18
    LABEL_FS  = 16
    TICK_FS   = 14
    LEGEND_FS = 13

    fig, ax = plt.subplots(figsize=(9.5, 6))

    ax.plot(Ks, y_opt_db, "o-", label="Optimal (max–min)")
    ax.plot(Ks, y_rzf_db, "s-", label="RZF (equal power)")
    ax.plot(Ks, y_zf_db,  "^-", label="ZF (equal power)")
    ax.plot(Ks, y_ex_db,  "d-", label="Exhaustive equal-power")

    ax.set_xlabel(r"Number of users $K$", fontsize=LABEL_FS)
    ax.set_ylabel("Balanced SINR (dB)", fontsize=LABEL_FS)
    ax.set_title(
        f"P1: Balanced SINR vs K (UMa)  |  M={M}, SNR={args.snr_db:.1f} dB, eta={args.eta}",
        fontsize=TITLE_FS
    )

    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)

    # ✅ Legend: right-middle (outside the plot to avoid hiding curves)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=LEGEND_FS,
        frameon=True
    )

    # ✅ Ensures room for the legend on the right
    fig.tight_layout()


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
