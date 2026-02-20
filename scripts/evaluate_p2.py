"""
P2 Evaluation Core (GPU-friendly, TensorFlow)
---------------------------------------------
Evaluate objective and feasibility for P2 given:
- channel tensor
- selected users per RB
- beamformers for selected users

This file intentionally contains the evaluation core only.
"""

from __future__ import annotations

import tensorflow as tf


@tf.function
def evaluate_p2(
    h_true: tf.Tensor,            # [B,K,N_RB,M]
    sel: tf.Tensor,               # [B,N_RB,U], -1 for empty
    W: tf.Tensor,                 # [B,N_RB,M,U]
    R_th: tf.Tensor,              # [K] or [B,K]
    sigma2: tf.Tensor,            # scalar
    B_RB: tf.Tensor,              # scalar
    P_tot: tf.Tensor,             # scalar
    U_max: int,
    rate_per_hz: bool = False,
    require_all_users: bool = False,
    r_max: int = 1,
    apply_channel_conjugate: bool = False,
):
    """
    Returns dict with:
      objective_power: [B]
      feasible: [B]
      feasible_rate: [B]
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

    # Gather selected channels
    h_nbkm = tf.transpose(h_true, [0, 2, 1, 3])              # [B,N_RB,K,M]
    h_sel = tf.gather(h_nbkm, sel_safe, axis=2, batch_dims=2) # [B,N_RB,U,M]
    h_sel = tf.where(valid[..., None], h_sel, tf.zeros_like(h_sel))

    # SINR for selected users
    h_eff = tf.math.conj(h_sel) if apply_channel_conjugate else h_sel
    HW = tf.einsum("bnim,bnmu->bniu", h_eff, W)             # [B,N_RB,U,U]
    sig = tf.abs(tf.linalg.diag_part(HW)) ** 2
    tot = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)
    interf = tf.maximum(tot - sig, 0.0)
    sinr_sel = sig / (interf + tf.cast(sigma2, sig.dtype))
    sinr_sel = tf.where(valid, sinr_sel, tf.zeros_like(sinr_sel))

    # Rates
    rates_core = tf.math.log(1.0 + sinr_sel) / tf.math.log(tf.constant(2.0, sinr_sel.dtype))
    rates_sel = rates_core if rate_per_hz else tf.cast(B_RB, rates_core.dtype) * rates_core

    # Required rates for selected users
    if tf.rank(R_th) == 1:
        R_bk = tf.broadcast_to(R_th[None, :], [B, K])
    else:
        R_bk = R_th
    R_nbk = tf.broadcast_to(R_bk[:, None, :], [B, tf.shape(sel)[1], K])
    R_req_sel = tf.gather(R_nbk, sel_safe, axis=2, batch_dims=2)
    R_req_sel = tf.where(valid, R_req_sel, tf.zeros_like(R_req_sel))

    # C1-rate
    c1_rate_sel = tf.where(valid, rates_sel >= tf.cast(R_req_sel, rates_sel.dtype), tf.ones_like(valid))

    # User assignment count across RBs
    onehot = tf.one_hot(sel_safe, depth=K, dtype=tf.int32)               # [B,N_RB,U,K]
    onehot = tf.where(valid[..., None], onehot, tf.zeros_like(onehot))
    assigned_count = tf.reduce_sum(onehot, axis=[1, 2])                  # [B,K]
    c1_rmax = assigned_count <= int(r_max)
    c1_served = assigned_count >= 1 if require_all_users else tf.ones_like(c1_rmax)

    # C2: users per RB
    users_per_rb = tf.reduce_sum(tf.cast(valid, tf.int32), axis=-1)      # [B,N_RB]
    c2 = users_per_rb <= int(U_max)

    # C3: total power
    p_user = tf.reduce_sum(tf.abs(W) ** 2, axis=2)                       # [B,N_RB,U]
    p_user = tf.where(valid, p_user, tf.zeros_like(p_user))
    p_rb = tf.reduce_sum(p_user, axis=-1)                                # [B,N_RB]
    p_tot = tf.reduce_sum(p_rb, axis=-1)                                 # [B]
    c3 = p_tot <= tf.cast(P_tot, p_tot.dtype)

    feasible_rate = tf.reduce_all(c1_rate_sel, axis=[1, 2]) & tf.reduce_all(c1_rmax, axis=1) & tf.reduce_all(c1_served, axis=1)
    feasible = feasible_rate & tf.reduce_all(c2, axis=1) & c3

    return {
        "objective_power": p_tot,
        "sinr_sel": sinr_sel,
        "rates_sel": rates_sel,
        "users_per_rb": users_per_rb,
        "assigned_count": assigned_count,
        "p_rb": p_rb,
        "p_tot": p_tot,
        "feasible_rate": feasible_rate,
        "feasible_c2": tf.reduce_all(c2, axis=1),
        "feasible_c3": c3,
        "feasible": feasible,
    }


__all__ = ["evaluate_p2"]
