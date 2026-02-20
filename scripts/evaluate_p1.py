"""
P1 Evaluation (GPU-friendly, TensorFlow)
---------------------------------------
Vectorized evaluation of the max-min SINR objective for a given
selection and beamformers, with feasibility checks for constraints.

Expected tensors:
  h_true: [B, K, N_RB, M]  complex64/complex128
  h_hat : [B, K, N_RB, M]  complex64/complex128 (optional)
  sel   : [B, N_RB, U_max] int32 (indices of users, -1 for empty)
  W     : [B, N_RB, M, U_max] complex (beamformers for selected users)
  rho   : [K] or [B, K] float (weights)
"""

from __future__ import annotations

import tensorflow as tf

from sim.sionna_channel import generate_channels


@tf.function
def evaluate_p1(
    h_true: tf.Tensor,
    sel: tf.Tensor,
    W: tf.Tensor,
    rho: tf.Tensor,
    sigma2: tf.Tensor,
    P_RB_max: tf.Tensor,
    P_tot: tf.Tensor,
    h_hat: tf.Tensor | None = None,
    use_h_hat_for_eval: bool = False,
    P_user_max: tf.Tensor | None = None,
    apply_channel_conjugate: bool = False,
):
    """
    Evaluate P1 objective and constraints.

    Returns dict with:
      - min_sinr_ratio_rb: [B, N_RB] min_k SINR_{k,n}/rho_k over selected users
      - min_sinr_ratio_all: [B] min over RBs (ignoring empty RBs)
      - feasible_c1: [B, N_RB] (|S_n| <= U_max) always true if sel is well-formed
      - feasible_c2_rb: [B, N_RB] (sum ||w||^2 <= P_RB_max)
      - feasible_c2_user: [B, N_RB, U_max] (||w_k||^2 <= P_user_max) if P_user_max provided
      - feasible_c3: [B] (sum_{n,k} ||w||^2 <= P_tot)
      - sinr: [B, N_RB, U_max] SINR for selected users (0 for invalid)
    """
    # Choose evaluation channel
    h_eval = h_hat if (use_h_hat_for_eval and h_hat is not None) else h_true

    # Shapes
    B = tf.shape(h_eval)[0]
    K = tf.shape(h_eval)[1]
    N_RB = tf.shape(h_eval)[2]
    U_max = tf.shape(sel)[2]

    # Masks for valid selections
    valid = sel >= 0  # [B, N_RB, U_max]
    valid_f = tf.cast(valid, tf.float32)

    # Clamp indices to valid range for gather
    sel_safe = tf.where(valid, sel, tf.zeros_like(sel))

    # Gather channels for selected users: [B, N_RB, U_max, M]
    # h_eval: [B, K, N_RB, M] -> transpose to [B, N_RB, K, M] for batch gather
    h_eval_nbkm = tf.transpose(h_eval, [0, 2, 1, 3])
    h_sel = tf.gather(h_eval_nbkm, sel_safe, axis=2, batch_dims=2)

    # Apply mask to h_sel to avoid garbage for invalid
    h_sel = tf.where(valid[..., None], h_sel, tf.zeros_like(h_sel))

    # Compute effective projections for all selected pairs: [B, N_RB, U_max, U_max]
    # For backward compatibility with existing scripts, default is H @ W (no explicit conjugation).
    h_eff = tf.math.conj(h_sel) if apply_channel_conjugate else h_sel
    HW = tf.einsum("bnim,bnmu->bniu", h_eff, W)

    # Signal power (diagonal)
    sig = tf.abs(tf.linalg.diag_part(HW)) ** 2  # [B, N_RB, U_max]
    # Total received power from all beams
    total = tf.reduce_sum(tf.abs(HW) ** 2, axis=-1)  # [B, N_RB, U_max]
    interf = tf.maximum(total - sig, 0.0)

    # Noise
    sigma2 = tf.cast(sigma2, sig.dtype)
    sinr = sig / (interf + sigma2)
    sinr = tf.where(valid, sinr, tf.zeros_like(sinr))

    # Gather rho for selected users
    if tf.rank(rho) == 1:
        rho_bk = tf.broadcast_to(rho[None, :], [B, K])
    else:
        rho_bk = rho
    rho_nbk = tf.transpose(rho_bk[:, None, :], [0, 1, 2])  # [B,1,K]
    rho_nbk = tf.broadcast_to(rho_nbk, [B, N_RB, K])
    rho_sel = tf.gather(rho_nbk, sel_safe, axis=2, batch_dims=2)  # [B,N_RB,U_max]
    rho_sel = tf.where(valid, rho_sel, tf.ones_like(rho_sel))

    sinr_ratio = sinr / tf.cast(rho_sel, sinr.dtype)

    # Min over selected users per RB 
    inf = tf.constant(1e30, dtype=sinr_ratio.dtype)
    sinr_ratio_masked = tf.where(valid, sinr_ratio, inf)
    min_sinr_ratio_rb = tf.reduce_min(sinr_ratio_masked, axis=-1)  # [B,N_RB]

    # Handle empty RBs: if no valid users, set to 0
    num_valid = tf.reduce_sum(tf.cast(valid, tf.int32), axis=-1)  # [B,N_RB]
    min_sinr_ratio_rb = tf.where(num_valid > 0, min_sinr_ratio_rb, tf.zeros_like(min_sinr_ratio_rb))

    # Global min across RBs (ignore empty RBs by treating them as +inf)
    inf_rb = tf.where(num_valid > 0, min_sinr_ratio_rb, inf * tf.ones_like(min_sinr_ratio_rb))
    min_sinr_ratio_all = tf.reduce_min(inf_rb, axis=-1)
    min_sinr_ratio_all = tf.where(
        tf.reduce_any(num_valid > 0, axis=-1), min_sinr_ratio_all, tf.zeros_like(min_sinr_ratio_all)
    )

    # Power constraints
    # per-user power (selected beams)
    p_user = tf.reduce_sum(tf.abs(W) ** 2, axis=2)  # [B,N_RB,U_max]
    p_user = tf.where(valid, p_user, tf.zeros_like(p_user))

    # C2 per RB: sum_k ||w||^2 <= P_RB_max
    p_rb = tf.reduce_sum(p_user, axis=-1)  # [B,N_RB]
    feasible_c2_rb = p_rb <= tf.cast(P_RB_max, p_rb.dtype)

    # Optional per-user cap
    if P_user_max is not None:
        feasible_c2_user = p_user <= tf.cast(P_user_max, p_user.dtype)
    else:
        feasible_c2_user = tf.ones_like(p_user, dtype=tf.bool)

    # C3 total power
    p_tot = tf.reduce_sum(p_rb, axis=-1)  # [B]
    feasible_c3 = p_tot <= tf.cast(P_tot, p_tot.dtype)

    # C1: number of selected users per RB <= U_max (always if sel is well-formed)
    feasible_c1 = num_valid <= U_max

    return {
        "min_sinr_ratio_rb": min_sinr_ratio_rb,
        "min_sinr_ratio_all": min_sinr_ratio_all,
        "sinr": sinr,
        "feasible_c1": feasible_c1,
        "feasible_c2_rb": feasible_c2_rb,
        "feasible_c2_user": feasible_c2_user,
        "feasible_c3": feasible_c3,
        "p_user": p_user,
        "p_rb": p_rb,
        "p_tot": p_tot,
    }


__all__ = ["evaluate_p1"]


@tf.function
def evaluate_p1_with_sionna(
    cfg,
    batch_size: int,
    sel: tf.Tensor,
    W: tf.Tensor,
    rho: tf.Tensor,
    use_h_hat_for_eval: bool = False,
    P_user_max: tf.Tensor | None = None,
):
    """
    Convenience wrapper:
      - generates (h_true, h_hat) using Sionna
      - calls evaluate_p1(...)
    """
    h_true, h_hat = generate_channels(cfg, batch_size)
    return evaluate_p1(
        h_true=h_true,
        h_hat=h_hat,
        sel=sel,
        W=W,
        rho=rho,
        sigma2=tf.constant(float(cfg.sigma2), tf.float32),
        P_RB_max=tf.constant(float(cfg.P_RB_max), tf.float32),
        P_tot=tf.constant(float(cfg.P_tot), tf.float32),
        use_h_hat_for_eval=use_h_hat_for_eval,
        P_user_max=P_user_max,
    )


__all__.append("evaluate_p1_with_sionna")
