# scripts/save_dataset_p123.py
# ----------------------------------------------------------
# Generates a dataset for P1/P2/P3 and saves:
#   - labels_p123.npz   (full Pythonic structure)
#   - summary_p123.csv  (simple CSV with sample-level objectives)
# ----------------------------------------------------------

import os
import csv
import numpy as np

from sim.config import SystemConfig
from sim.gen_labels_p123 import generate_dataset


def main():
    cfg = SystemConfig()
    N_samples = 2000
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {N_samples} samples...")
    samples = generate_dataset(cfg, N_samples=N_samples, sinr_target_dB=5.0)

    # ---------- 1) Save NPZ (Pythonic, full structure) ----------
    np.savez(
        os.path.join(out_dir, "labels_p123.npz"),
        samples=samples,
        allow_pickle=True,
    )
    print("Saved NPZ:", os.path.join(out_dir, "labels_p123.npz"))

    # ---------- 2) Simple CSV summary (per sample) ----------
    csv_path = os.path.join(out_dir, "summary_p123.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "P1_gamma_min_dB",
            "P1_num_users",
            "P2_total_power",
            "P2_num_users",
            "P3_sum_rate_bits",
            "P3_num_users",
        ])

        for idx, s in enumerate(samples):
            probs = s["problems"]

            # P1
            P1_obj = probs["P1"]["objective"]
            P1_users = probs["P1"]["RB_assignment"][0].shape[0]

            # P2
            P2_obj = probs["P2"]["objective"]
            P2_users = probs["P2"]["RB_assignment"][0].shape[0]

            # P3
            P3_obj = probs["P3"]["objective"]
            P3_users = probs["P3"]["RB_assignment"][0].shape[0]

            writer.writerow([
                idx,
                P1_obj,
                P1_users,
                P2_obj,
                P2_users,
                P3_obj,
                P3_users,
            ])

    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    main()
