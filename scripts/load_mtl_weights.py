from __future__ import annotations

import argparse
import os

import tensorflow as tf

try:
    from config import SystemConfig
    from sim.mtl_unsupervised import MTLUnsupervisedModel
except ImportError:
    import sys

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from config import SystemConfig
    from sim.mtl_unsupervised import MTLUnsupervisedModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="results/mtl_unsupervised.weights.h5")
    args = ap.parse_args()

    cfg = SystemConfig()
    model = MTLUnsupervisedModel(
        K=cfg.K,
        N_RB=cfg.N_RB,
        M=cfg.M,
        U_max=min(cfg.U_max, cfg.M),
        P_RB_max=cfg.P_RB_max,
        P_tot=cfg.P_tot,
    )

    # Build model once before loading weights.
    dummy = tf.complex(
        tf.zeros([1, cfg.K, cfg.N_RB, cfg.M], tf.float32),
        tf.zeros([1, cfg.K, cfg.N_RB, cfg.M], tf.float32),
    )
    _ = model(dummy, training=False)

    model.load_weights(args.weights)
    print(f"Loaded weights from: {args.weights}")


if __name__ == "__main__":
    main()
