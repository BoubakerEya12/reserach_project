"""
Train an unsupervised multi-task model for:
  - scheduling
  - beamforming
  - power allocation

Default data source: sim.sionna_channel.generate_channels
Fallback: synthetic Rayleigh if Sionna is unavailable.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import tensorflow as tf

try:
    from config import SystemConfig
    from sim.mtl_unsupervised import MTLTrainConfig, MTLUnsupervisedModel, train_step
except ImportError:
    import sys

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from config import SystemConfig
    from sim.mtl_unsupervised import MTLTrainConfig, MTLUnsupervisedModel, train_step

try:
    from sim.sionna_channel import generate_channels
    HAS_SIONNA = True
except Exception:
    HAS_SIONNA = False


def _complex_normal(shape):
    re = tf.random.normal(shape, dtype=tf.float32)
    im = tf.random.normal(shape, dtype=tf.float32)
    return tf.complex(re, im) / tf.cast(tf.sqrt(2.0), tf.complex64)


def _fallback_channels(cfg: SystemConfig, batch_size: int):
    h_true = _complex_normal([batch_size, cfg.K, cfg.N_RB, cfg.M])
    eta = float(cfg.eta)
    e = _complex_normal([batch_size, cfg.K, cfg.N_RB, cfg.M])
    h_hat = tf.cast(eta, tf.complex64) * h_true + tf.cast(np.sqrt(max(1.0 - eta * eta, 0.0)), tf.complex64) * e
    return h_true, h_hat


def _to_float(x):
    return float(x.numpy()) if hasattr(x, "numpy") else float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--w_sched", type=float, default=1.0)
    ap.add_argument("--w_beam", type=float, default=1.0)
    ap.add_argument("--w_power", type=float, default=1.0)
    ap.add_argument("--entropy_weight", type=float, default=0.01)
    ap.add_argument("--unscheduled_weight", type=float, default=1.0)
    ap.add_argument("--leakage_weight", type=float, default=0.1)
    ap.add_argument("--qos_weight", type=float, default=2.0)

    ap.add_argument("--rate_target_bps", type=float, default=None)
    ap.add_argument("--gamma_qos_db", type=float, default=5.0)
    ap.add_argument("--init_weights", type=str, default="")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--save_every_steps", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default="results")
    args = ap.parse_args()

    cfg = SystemConfig()
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = MTLUnsupervisedModel(
        K=cfg.K,
        N_RB=cfg.N_RB,
        M=cfg.M,
        U_max=min(cfg.U_max, cfg.M),
        P_RB_max=cfg.P_RB_max,
        P_tot=cfg.P_tot,
        hidden_dim=128,
        depth=2,
    )

    train_cfg = MTLTrainConfig(
        w_sched=args.w_sched,
        w_beam=args.w_beam,
        w_power=args.w_power,
        entropy_weight=args.entropy_weight,
        unscheduled_weight=args.unscheduled_weight,
        leakage_weight=args.leakage_weight,
        qos_weight=args.qos_weight,
    )

    # Build once (required before loading optional initial weights).
    dummy = tf.complex(
        tf.zeros([1, cfg.K, cfg.N_RB, cfg.M], tf.float32),
        tf.zeros([1, cfg.K, cfg.N_RB, cfg.M], tf.float32),
    )
    _ = model(dummy, training=False)
    if args.init_weights:
        model.load_weights(args.init_weights)
        print(f"Initialized from weights: {args.init_weights}")

    if args.rate_target_bps is None:
        gamma_lin = float(10.0 ** (args.gamma_qos_db / 10.0))
        rate_target_bps = float(cfg.B_RB * np.log2(1.0 + gamma_lin))
    else:
        rate_target_bps = float(args.rate_target_bps)

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    print("==== Unsupervised MTL Training ====")
    print(f"Sionna available: {HAS_SIONNA}")
    print(f"K={cfg.K}, M={cfg.M}, N_RB={cfg.N_RB}, U_max={cfg.U_max}, R_max={cfg.R_max}")
    print(f"P_tot={cfg.P_tot:.4f}, P_RB_max={cfg.P_RB_max:.4f}, sigma2={cfg.sigma2:.3e}")
    print(f"QoS target rate={rate_target_bps:.3f} bps/user, gamma_ref={args.gamma_qos_db:.2f} dB")
    print(
        "Loss weights:"
        f" w_sched={train_cfg.w_sched:.3f},"
        f" w_beam={train_cfg.w_beam:.3f},"
        f" w_power={train_cfg.w_power:.3f},"
        f" qos_weight={train_cfg.qos_weight:.3f}"
    )

    history = []
    os.makedirs(args.save_dir, exist_ok=True)
    last_weight_path = os.path.join(args.save_dir, "mtl_unsupervised.last.weights.h5")
    csv_path = os.path.join(args.save_dir, "mtl_unsupervised_history.csv")
    for ep in range(1, args.epochs + 1):
        meter = {
            "loss_total": 0.0,
            "loss_sched": 0.0,
            "loss_beam": 0.0,
            "loss_power": 0.0,
            "mean_sum_rate": 0.0,
            "mean_user_rate": 0.0,
            "qos_user_ok_frac": 0.0,
            "mean_p_tot": 0.0,
            "mean_assign": 0.0,
            "mean_load": 0.0,
        }

        for step_idx in range(1, args.steps_per_epoch + 1):
            if HAS_SIONNA:
                h_true, h_hat = generate_channels(cfg, args.batch_size)
            else:
                h_true, h_hat = _fallback_channels(cfg, args.batch_size)

            logs = train_step(
                model=model,
                optimizer=opt,
                h_true=tf.cast(h_true, tf.complex64),
                h_hat=tf.cast(h_hat, tf.complex64),
                sigma2=float(cfg.sigma2),
                B_RB=float(cfg.B_RB),
                R_max=int(cfg.R_max),
                train_cfg=train_cfg,
                rate_target_bps=rate_target_bps,
            )

            for k in meter:
                meter[k] += _to_float(logs[k])

            if int(args.save_every_steps) > 0 and (step_idx % int(args.save_every_steps) == 0):
                model.save_weights(last_weight_path)
                print(
                    f"Step-checkpoint saved: {last_weight_path} "
                    f"(epoch={ep}, step={step_idx}/{args.steps_per_epoch})"
                )

        for k in meter:
            meter[k] /= float(args.steps_per_epoch)
        history.append([ep] + [meter[k] for k in meter.keys()])

        print(
            f"[epoch {ep:03d}] "
            f"L={meter['loss_total']:.4f} "
            f"(S={meter['loss_sched']:.4f}, B={meter['loss_beam']:.4f}, P={meter['loss_power']:.4f}) "
            f"SR={meter['mean_sum_rate']:.2f} "
            f"Ruser={meter['mean_user_rate']:.2f} "
            f"QoS_ok={meter['qos_user_ok_frac']:.4f} "
            f"Ptot={meter['mean_p_tot']:.4f}"
        )

        if int(args.save_every) > 0 and (ep % int(args.save_every) == 0):
            model.save_weights(last_weight_path)
            np.savetxt(
                csv_path,
                np.array(history, dtype=np.float64),
                delimiter=",",
                header=",".join(
                    [
                        "epoch",
                        "loss_total",
                        "loss_sched",
                        "loss_beam",
                        "loss_power",
                        "mean_sum_rate",
                        "mean_user_rate",
                        "qos_user_ok_frac",
                        "mean_p_tot",
                        "mean_assign",
                        "mean_load",
                    ]
                ),
                comments="",
            )
            print(f"Checkpoint saved: {last_weight_path}")

    weight_path = os.path.join(args.save_dir, "mtl_unsupervised.weights.h5")

    model.save_weights(weight_path)

    header = [
        "epoch",
        "loss_total",
        "loss_sched",
        "loss_beam",
        "loss_power",
        "mean_sum_rate",
        "mean_user_rate",
        "qos_user_ok_frac",
        "mean_p_tot",
        "mean_assign",
        "mean_load",
    ]
    np.savetxt(csv_path, np.array(history, dtype=np.float64), delimiter=",", header=",".join(header), comments="")

    print(f"Saved weights: {weight_path}")
    print(f"Saved history: {csv_path}")


if __name__ == "__main__":
    main()
