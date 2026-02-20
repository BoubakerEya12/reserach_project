"""
Compare evaluate_p1 outputs on CPU vs GPU (small batch).
Run:
  python -m scripts.check_p1_gpu_cpu
"""
import numpy as np
import tensorflow as tf

from scripts.evaluate_p1 import evaluate_p1


def main():
    # Small random test
    B, K, N_RB, M, U_max = 2, 6, 3, 4, 2
    rng = np.random.default_rng(0)

    h_re = rng.standard_normal((B, K, N_RB, M)).astype(np.float32)
    h_im = rng.standard_normal((B, K, N_RB, M)).astype(np.float32)
    h_true = h_re + 1j * h_im

    # Random selection (no -1)
    sel = rng.integers(0, K, size=(B, N_RB, U_max), dtype=np.int32)

    # Random beamformers
    W_re = rng.standard_normal((B, N_RB, M, U_max)).astype(np.float32)
    W_im = rng.standard_normal((B, N_RB, M, U_max)).astype(np.float32)
    W = W_re + 1j * W_im

    rho = np.ones((K,), dtype=np.float32)

    sigma2 = tf.constant(1e-3, tf.float32)
    P_RB_max = tf.constant(1e3, tf.float32)
    P_tot = tf.constant(1e6, tf.float32)

    h_true_tf = tf.constant(h_true)
    sel_tf = tf.constant(sel)
    W_tf = tf.constant(W)
    rho_tf = tf.constant(rho)

    # CPU eval
    with tf.device("/CPU:0"):
        out_cpu = evaluate_p1(
            h_true=h_true_tf,
            sel=sel_tf,
            W=W_tf,
            rho=rho_tf,
            sigma2=sigma2,
            P_RB_max=P_RB_max,
            P_tot=P_tot,
        )

    # GPU eval (if available)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected; CPU result only.")
        return

    with tf.device("/GPU:0"):
        out_gpu = evaluate_p1(
            h_true=h_true_tf,
            sel=sel_tf,
            W=W_tf,
            rho=rho_tf,
            sigma2=sigma2,
            P_RB_max=P_RB_max,
            P_tot=P_tot,
        )

    # Compare a few key tensors
    def max_abs_diff(a, b):
        return float(np.max(np.abs(a - b)))

    cpu_rb = out_cpu["min_sinr_ratio_rb"].numpy()
    gpu_rb = out_gpu["min_sinr_ratio_rb"].numpy()
    cpu_all = out_cpu["min_sinr_ratio_all"].numpy()
    gpu_all = out_gpu["min_sinr_ratio_all"].numpy()

    print("max |min_sinr_ratio_rb cpu-gpu| =", max_abs_diff(cpu_rb, gpu_rb))
    print("max |min_sinr_ratio_all cpu-gpu| =", max_abs_diff(cpu_all, gpu_all))


if __name__ == "__main__":
    main()
