from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np


def _read_p2_csv(path: str) -> Dict[str, np.ndarray]:
    gamma = []
    p_opt = []
    p_zf = []
    infeas_opt = []
    infeas_zf = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            gamma.append(float(row["gamma_dB"]))
            p_opt.append(float(row["P2_opt_avg_power"]))
            p_zf.append(float(row["P2_zf_avg_power"]))
            infeas_opt.append(float(row["P2_opt_infeasible_count"]))
            infeas_zf.append(float(row["P2_zf_infeasible_count"]))
    return {
        "gamma_dB": np.array(gamma, dtype=np.float64),
        "p_opt": np.array(p_opt, dtype=np.float64),
        "p_zf": np.array(p_zf, dtype=np.float64),
        "infeas_opt": np.array(infeas_opt, dtype=np.float64),
        "infeas_zf": np.array(infeas_zf, dtype=np.float64),
    }


def _read_metric_value_csv(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[str(row["metric"])] = float(row["value"])
    return out


def _nearest_gamma_idx(arr: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(arr - float(target))))


def _save_summary(path: str, rows: List[tuple[str, float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in rows:
            w.writerow([k, v])


def _make_figure(
    out_path: str,
    p2: Dict[str, np.ndarray],
    mtl: Dict[str, float],
    gamma_ref: float,
    idx_ref: int,
) -> None:
    import matplotlib.pyplot as plt

    g = p2["gamma_dB"]
    p_opt = p2["p_opt"]
    p_zf = p2["p_zf"]

    mtl_p = mtl["mean_total_power_w"]
    mtl_hard_rate = mtl["mean_hard_sum_rate_bps"]
    mtl_feas = mtl["frac_all_ok"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(g, p_opt, "o-", label="P2 optimal")
    ax[0].plot(g, p_zf, "s-", label="P2 ZF heuristic")
    ax[0].axhline(mtl_p, color="tab:green", linestyle="--", label=f"MTL mean power={mtl_p:.3f} W")
    ax[0].set_xlabel("Target SINR Γ (dB)")
    ax[0].set_ylabel("Average total power (W)")
    ax[0].set_title("Power comparison")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    labels = ["P2-opt", "P2-ZF", "MTL"]
    values = [p_opt[idx_ref], p_zf[idx_ref], mtl_p]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax[1].bar(labels, values, color=colors, alpha=0.9)
    ax[1].set_ylabel("Power (W)")
    ax[1].set_title(
        f"At Γ={gamma_ref:.1f} dB\n"
        f"MTL hard rate={mtl_hard_rate:.1f} bps, feasibility={mtl_feas:.3f}"
    )
    ax[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2_csv", type=str, default="results/results_p2_compare.csv")
    ap.add_argument("--mtl_summary_csv", type=str, default="results/mtl_eval_summary.csv")
    ap.add_argument("--gamma_ref", type=float, default=5.0)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    p2 = _read_p2_csv(args.p2_csv)
    mtl = _read_metric_value_csv(args.mtl_summary_csv)

    idx_ref = _nearest_gamma_idx(p2["gamma_dB"], args.gamma_ref)
    gamma_ref = float(p2["gamma_dB"][idx_ref])

    summary_rows = [
        ("gamma_ref_db", gamma_ref),
        ("p2_opt_power_at_gamma_ref_w", float(p2["p_opt"][idx_ref])),
        ("p2_zf_power_at_gamma_ref_w", float(p2["p_zf"][idx_ref])),
        ("mtl_mean_total_power_w", float(mtl["mean_total_power_w"])),
        ("mtl_mean_hard_sum_rate_bps", float(mtl["mean_hard_sum_rate_bps"])),
        ("mtl_frac_all_ok", float(mtl["frac_all_ok"])),
        ("ratio_mtl_over_p2_opt", float(mtl["mean_total_power_w"] / max(p2["p_opt"][idx_ref], 1e-12))),
        ("ratio_mtl_over_p2_zf", float(mtl["mean_total_power_w"] / max(p2["p_zf"][idx_ref], 1e-12))),
    ]
    if "qos_gamma_db" in mtl:
        summary_rows.append(("qos_gamma_db", float(mtl["qos_gamma_db"])))
    if "qos_user_rate_ok_frac" in mtl:
        summary_rows.append(("qos_user_rate_ok_frac", float(mtl["qos_user_rate_ok_frac"])))
    if "qos_user_sinr_ok_frac" in mtl:
        summary_rows.append(("qos_user_sinr_ok_frac", float(mtl["qos_user_sinr_ok_frac"])))
    if "qos_sample_all_users_rate_ok_frac" in mtl:
        summary_rows.append(("qos_sample_all_users_rate_ok_frac", float(mtl["qos_sample_all_users_rate_ok_frac"])))
    if "qos_sample_all_users_sinr_ok_frac" in mtl:
        summary_rows.append(("qos_sample_all_users_sinr_ok_frac", float(mtl["qos_sample_all_users_sinr_ok_frac"])))

    out_summary = os.path.join(args.out_dir, "mtl_vs_baseline_summary.csv")
    out_fig = os.path.join(args.out_dir, "mtl_vs_baseline.png")
    _save_summary(out_summary, summary_rows)
    figure_saved = True
    try:
        _make_figure(out_fig, p2, mtl, gamma_ref, idx_ref)
    except ModuleNotFoundError:
        figure_saved = False

    print("Saved:", out_summary)
    if figure_saved:
        print("Saved:", out_fig)
    else:
        print("Figure skipped: matplotlib is not installed in this environment")
    print("Reference gamma (dB):", gamma_ref)
    print("P2-opt @gamma (W):", float(p2["p_opt"][idx_ref]))
    print("P2-ZF  @gamma (W):", float(p2["p_zf"][idx_ref]))
    print("MTL mean power (W):", float(mtl["mean_total_power_w"]))
    print("MTL mean hard sum-rate (bps):", float(mtl["mean_hard_sum_rate_bps"]))
    print("MTL feasibility:", float(mtl["frac_all_ok"]))
    if "qos_user_rate_ok_frac" in mtl:
        print("MTL QoS user-rate satisfaction:", float(mtl["qos_user_rate_ok_frac"]))
    if "qos_user_sinr_ok_frac" in mtl:
        print("MTL QoS user-SINR satisfaction:", float(mtl["qos_user_sinr_ok_frac"]))
    if "qos_sample_all_users_rate_ok_frac" in mtl:
        print("MTL QoS all-users-rate satisfaction:", float(mtl["qos_sample_all_users_rate_ok_frac"]))
    if "qos_sample_all_users_sinr_ok_frac" in mtl:
        print("MTL QoS all-users-SINR satisfaction:", float(mtl["qos_sample_all_users_sinr_ok_frac"]))


if __name__ == "__main__":
    main()
