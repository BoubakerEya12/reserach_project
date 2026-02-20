"""
P3 Evaluation Core (GPU-friendly, TensorFlow)
---------------------------------------------
Evaluate objective and feasibility for P3 given:
- channel tensor
- selected users per RB
- beamformers for selected users
- user weights alpha
"""

from __future__ import annotations

import tensorflow as tf


@tf.function
def evaluate_p3(
    h_true: tf.Tensor,            # [B,K,N_RB,M]
    sel: tf.Tensor,               # [B,N_RB,U], -1 for empty
    W: tf.Tensor,                 # [B,N_RB,M,U]
    alpha: tf.Tensor,             # [K] or [B,K]
    sigma2: tf.Tensor,            # scalar
    P_RB_max: tf.Tensor,          # scalar
    P_tot: tf.Tensor,             # scalar
    U_max: int,
    apply_channel_conjugate: bool = False,
    power_tol: float = 1e-5,
):
    """
    Returns dict with:
      objective_wsr: [B]
      feasible: [B]
      feasible_c1: [B]
      feasible_c2: [B]
      feasible_c3: [B]
      sinr_sel: [B,N_RB,U]
      rates_sel: [B,N_RB,U]
      p_rb: [B,N_RB]
      p_tot: [B]
    """
    B = tf.shape(h_true)[0]
    K = tf.shape(h_true)[1]

    valid = sel >= 0
    sel_safe = tf.where(valid, sel, tf.zeros_like(sel))

    # [B,K,N_RB,M] -> [B,N_RB,K,M]
    h_nbkm = tf.transpose(h_true, [0, 2, 1, 3])
    h_sel = tf.gather(h_nbkm, sel_safe, axis=2, batch_dims=2)  # [B,N_RB,U,M]
    h_sel = tf.where(valid[..., None], h_sel, tf.zeros_like(h_sel))

    # SINR for selected users
    h_eff = tf.math.conj(h_sel) if apply_channel_conjugate else h_sel
    HW = tf.einsum("bnim,bnmu->bniu", h_eff, W)              # [B,N_RB,U,U]
    sig = tf.abs(tf.linalg.diag_part(HW)) ** 2
    tot = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interf = tf.maximum(tot - sig, 0.0)
    sinr_sel = sig / (interf + tf.cast(sigma2, sig.dtype))
    sinr_sel = tf.where(valid, sinr_sel, tf.zeros_like(sinr_sel))

    rates_sel = tf.math.log(1.0 + sinr_sel) / tf.math.log(tf.constant(2.0, sinr_sel.dtype))

    # Gather alpha for selected users.
    # Robust handling for alpha as [K] or [B,K] inside tf.function.
    alpha_2d = tf.reshape(alpha, [-1, K])      # [1,K] or [B,K]
    alpha_bk = tf.broadcast_to(alpha_2d, [B, K])
    alpha_nbk = tf.broadcast_to(alpha_bk[:, None, :], [B, tf.shape(sel)[1], K])
    alpha_sel = tf.gather(alpha_nbk, sel_safe, axis=2, batch_dims=2)
    alpha_sel = tf.where(valid, alpha_sel, tf.zeros_like(alpha_sel))

    # Objective: sum_n sum_{k in S_n} alpha_k log2(1+sinr_k,n)
    objective_wsr = tf.reduce_sum(tf.cast(alpha_sel, rates_sel.dtype) * rates_sel, axis=[1, 2])

    # C1: |S_n| <= min(Nt, Umax) ; Nt handled by caller using U_max=min(cfg.U_max,cfg.M)
    users_per_rb = tf.reduce_sum(tf.cast(valid, tf.int32), axis=-1)      # [B,N_RB]
    c1 = users_per_rb <= int(U_max)

    # C2: per-RB power <= P_RB_max
    p_user = tf.reduce_sum(tf.abs(W) ** 2, axis=2)                        # [B,N_RB,U]
    p_user = tf.where(valid, p_user, tf.zeros_like(p_user))
    p_rb = tf.reduce_sum(p_user, axis=-1)                                 # [B,N_RB]
    c2 = p_rb <= (tf.cast(P_RB_max, p_rb.dtype) + tf.cast(power_tol, p_rb.dtype))

    # C3: total power <= P_tot
    p_tot = tf.reduce_sum(p_rb, axis=-1)                                  # [B]
    c3 = p_tot <= (tf.cast(P_tot, p_tot.dtype) + tf.cast(power_tol, p_tot.dtype))

    feasible = tf.reduce_all(c1, axis=1) & tf.reduce_all(c2, axis=1) & c3

    return {
        "objective_wsr": objective_wsr,
        "sinr_sel": sinr_sel,
        "rates_sel": rates_sel,
        "users_per_rb": users_per_rb,
        "p_rb": p_rb,
        "p_tot": p_tot,
        "feasible_c1": tf.reduce_all(c1, axis=1),
        "feasible_c2": tf.reduce_all(c2, axis=1),
        "feasible_c3": c3,
        "feasible": feasible,
    }


__all__ = ["evaluate_p3"]
