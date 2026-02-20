# scripts/plot_p3_wmmse_only.py
# Exemple :
#   python -m scripts.plot_p3_wmmse_only --n_slots 1000

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

    parser.add_argument("--out_csv", type=str, default="results/p3_wmmse_vs_snr.csv")
    parser.add_argument("--out_fig", type=str, default="results/p3_wmmse_vs_snr.png")
    args = parser.parse_args()

    cfg = SystemConfig()
    snr_grid = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)

    mode = "wmmse_alpha_beta"
    label = "WMMSE (greedy on alpha·beta)"

    print("=== P3 WSR vs SNR (WMMSE only) ===")
    print(f"M={cfg.M}, K={cfg.K}, N_RB={cfg.N_RB}, U_max={cfg.U_max}")
    print(f"Per-RB power cap P_RB_max = {cfg.P_RB_max:.2f} W")
    print(f"Mode: {mode}")
    print(f"SNR grid: {snr_grid} dB")
    print(f"Monte-Carlo: {args.n_slots} slots par point\n")

    mean_wsr = np.zeros(len(snr_grid), dtype=np.float64)
    std_wsr = np.zeros_like(mean_wsr)

    for i_snr, snr_db in enumerate(snr_grid):
        print(f"SNR = {snr_db:.1f} dB")
        mu, sig = evaluate_p3_wsr_at_snr(
            cfg,
            snr_db=snr_db,
            n_slots=args.n_slots,
            seed=args.seed + i_snr * 10,
            mode=mode,
        )
        mean_wsr[i_snr] = mu
        std_wsr[i_snr] = sig
        print(f"  {label:<30s} -> mean = {mu:7.3f}, std = {sig:7.3f}\n")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    out_csv_dir = os.path.dirname(args.out_csv)
    if out_csv_dir:  # avoid error if user gives filename without folder
        os.makedirs(out_csv_dir, exist_ok=True)

    header = "snr_db,wmmse_mean,wmmse_std"
    rows = np.column_stack([snr_grid, mean_wsr, std_wsr])
    np.savetxt(args.out_csv, rows, delimiter=",", header=header, comments="")
    print(f"CSV sauvegardé dans : {args.out_csv}")

    # ------------------------------------------------------------------
    # Plot figure
    # ------------------------------------------------------------------
    out_fig_dir = os.path.dirname(args.out_fig)
    if out_fig_dir:
        os.makedirs(out_fig_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        snr_grid,
        mean_wsr,
        yerr=std_wsr,
        marker="o",
        linestyle="-",
        capsize=3,
        label=label,
    )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Weighted Sum Rate (bit/s/Hz)")
    plt.title("P3 – WSR vs SNR (WMMSE only, Sionna UMa)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)
    print(f"Figure sauvegardée dans : {args.out_fig}")


if __name__ == "__main__":
    main()
