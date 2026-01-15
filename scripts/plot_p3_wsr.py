# scripts/plot_p3_wsr.py
# python -m scripts.plot_p3_wsr --n_slots 1000
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from config import SystemConfig
from sim.p3_wsr_eval import evaluate_p3_wsr_at_snr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=30.0)
    parser.add_argument("--snr_step", type=float, default=5.0)
    parser.add_argument("--n_slots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_csv", type=str, default="results/p3_wsr_vs_snr.csv")
    parser.add_argument("--out_fig", type=str, default="results/p3_wsr_vs_snr.png")
    args = parser.parse_args()

    cfg = SystemConfig()

    snr_grid = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
    print("=== P3 WSR vs SNR (multiple methods) ===")
    print(f"M={cfg.M}, K={cfg.K}, N_RB={cfg.N_RB}, U_max={cfg.U_max}")
    print(f"Per-RB power cap P_RB_max = {cfg.P_RB_max:.2f} W")
    print(f"SNR grid: {snr_grid} dB")
    print(f"Monte-Carlo: {args.n_slots} slots par point\n")

    # Méthodes comparées
    methods = [
        ("zf_beta",          "ZF (greedy on beta)"),
        ("zf_random",        "ZF (random users)"),
        ("wmmse_alpha_beta", "WMMSE (greedy on alpha·beta)"),
        ("mrt_single",       "Single-user MRT (upper bound)"),
    ]

    n_methods = len(methods)
    mean_wsr = np.zeros((n_methods, len(snr_grid)), dtype=np.float64)
    std_wsr = np.zeros_like(mean_wsr)

    for i_snr, snr_db in enumerate(snr_grid):
        print(f"SNR = {snr_db:.1f} dB")
        for m, (mode, label) in enumerate(methods):
            mu, sig = evaluate_p3_wsr_at_snr(
                cfg,
                snr_db=snr_db,
                n_slots=args.n_slots,
                seed=args.seed + i_snr * 10 + m,
                mode=mode,
            )
            mean_wsr[m, i_snr] = mu
            std_wsr[m, i_snr] = sig
            print(f"  {label:<30s} -> mean = {mu:7.3f}, std = {sig:7.3f}")
        print()

    # ------------------------------------------------------------------
    # Sauvegarde CSV
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = "snr_db," + ",".join(
        [f"{name}_mean,{name}_std" for name, _ in methods]
    )
    rows = []
    for i in range(len(snr_grid)):
        row = [snr_grid[i]]
        for m in range(n_methods):
            row.append(mean_wsr[m, i])
            row.append(std_wsr[m, i])
        rows.append(row)
    rows = np.array(rows, dtype=np.float64)
    np.savetxt(args.out_csv, rows, delimiter=",", header=header, comments="")
    print(f"CSV sauvegardé dans : {args.out_csv}")

    # ------------------------------------------------------------------
    # Trace la figure
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)

    plt.figure(figsize=(7, 5))
    markers = ["o", "s", "^", "d"]
    linestyles = ["-", "--", "-.", ":"]
    for m, (mode, label) in enumerate(methods):
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
    plt.title("P3 – WSR vs SNR (3GPP UMa, alpha(beta))")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)
    print(f"Figure sauvegardée dans : {args.out_fig}")


if __name__ == "__main__":
    main()
