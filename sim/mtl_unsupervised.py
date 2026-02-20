from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class MTLTrainConfig:
    """Hyper-parameters for unsupervised MTL training."""

    w_sched: float = 1.0
    w_beam: float = 1.0
    w_power: float = 1.0

    entropy_weight: float = 0.01
    unscheduled_weight: float = 1.0
    leakage_weight: float = 0.1
    qos_weight: float = 2.0

    eps: float = 1e-9


class MTLUnsupervisedModel(tf.keras.Model):
    """
    One-pass MTL model for:
      1) user-RB scheduling (soft assignment)
      2) unit-norm beam directions
      3) power split under BS constraints

    Input:
      h_hat: [B, K, N_RB, M] complex

    Outputs:
      q:     [B, K, N_RB] soft scheduling weights in [0,1]
      v_re:  [B, K, N_RB, M] real part of unit-norm beam directions
      v_im:  [B, K, N_RB, M] imag part of unit-norm beam directions
      p:     [B, K, N_RB] nonnegative power, projected to satisfy caps
      s:     [B, K, N_RB] effective scheduled power = q * p
      q_all: [B, K, N_RB+1] includes "not scheduled" column
    """

    def __init__(
        self,
        K: int,
        N_RB: int,
        M: int,
        U_max: int,
        P_RB_max: float,
        P_tot: float,
        hidden_dim: int = 128,
        depth: int = 2,
    ):
        super().__init__()
        self.K = int(K)
        self.N_RB = int(N_RB)
        self.M = int(M)
        self.U_max = int(U_max)
        self.P_RB_max = float(P_RB_max)
        self.P_tot = float(P_tot)

        self.shared = []
        for _ in range(int(depth)):
            self.shared.append(tf.keras.layers.Dense(hidden_dim, activation="relu"))

        # Scheduling head (per user, across RB + one "not scheduled" class)
        self.sched_head = tf.keras.layers.Dense(self.N_RB + 1, activation=None)

        # Beam head (per user-RB): 2M reals -> complex M-vector
        self.beam_head = tf.keras.layers.Dense(2 * self.M, activation=None)

        # Power head (per user-RB): one scalar logit
        self.power_head = tf.keras.layers.Dense(1, activation=None)

    def _features(self, h_hat: tf.Tensor) -> tf.Tensor:
        # [B,K,N,M] complex -> real features [B,K,N,3M]
        re = tf.math.real(h_hat)
        im = tf.math.imag(h_hat)
        mag = tf.abs(h_hat)
        return tf.concat([re, im, mag], axis=-1)

    def call(self, h_hat: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        x = self._features(h_hat)

        # Shared trunk on each (k,n)
        for layer in self.shared:
            x = layer(x, training=training)

        # ---------- Scheduling ----------
        # Aggregate per user across RB to make one RB choice per user.
        x_user = tf.reshape(x, [tf.shape(x)[0], self.K, self.N_RB * tf.shape(x)[-1]])
        sched_logits = self.sched_head(x_user, training=training)            # [B,K,N_RB+1]
        q_all = tf.nn.softmax(sched_logits, axis=-1)
        q = q_all[..., : self.N_RB]

        # ---------- Beam directions ----------
        beam_raw = self.beam_head(x, training=training)                      # [B,K,N_RB,2M]
        v_re = beam_raw[..., : self.M]
        v_im = beam_raw[..., self.M :]
        norm = tf.sqrt(tf.reduce_sum(v_re * v_re + v_im * v_im, axis=-1, keepdims=True) + 1e-9)
        v_re = v_re / norm
        v_im = v_im / norm

        # ---------- Power split ----------
        p_logits = self.power_head(x, training=training)[..., 0]             # [B,K,N_RB]
        p_score = tf.nn.softplus(p_logits) + 1e-9

        # Important: do not attenuate by q here because s=q*p is already applied
        # later. Multiplying by q twice (here and in s) makes effective power
        # collapse and hurts QoS severely.
        p_score = p_score + 1e-9

        # Per-RB projection: sum_k p_{k,n} <= P_RB_max
        rb_den = tf.reduce_sum(p_score, axis=1, keepdims=True) + 1e-9        # [B,1,N_RB]
        p_rb_feasible = p_score / rb_den * self.P_RB_max                     # [B,K,N_RB]

        # Global projection: sum_{k,n} p_{k,n} <= P_tot
        p_tot = tf.reduce_sum(p_rb_feasible, axis=[1, 2], keepdims=True)
        scale = tf.minimum(1.0, self.P_tot / (p_tot + 1e-9))
        p = p_rb_feasible * scale

        s = q * p

        return {
            "q": q,
            "q_all": q_all,
            "v_re": v_re,
            "v_im": v_im,
            "p": p,
            "s": s,
        }


def _sinr_and_rates(
    h_true: tf.Tensor,
    v_re_in: tf.Tensor,
    v_im_in: tf.Tensor,
    s: tf.Tensor,
    sigma2: float,
    B_RB: float,
    eps: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute SINR and rates under soft scheduling/power.

    Args:
      h_true: [B,K,N_RB,M] complex
      v_re_in: [B,K,N_RB,M] float
      v_im_in: [B,K,N_RB,M] float
      s:      [B,K,N_RB] float, effective allocated power

    Returns:
      sinr:      [B,K,N_RB]
      rate_kn:   [B,K,N_RB]
      user_rate: [B,K]
      p_tot:     [B]
    """
    # Put RB first for easier multi-user interference computation.
    h_bnkm = tf.transpose(h_true, [0, 2, 1, 3])  # [B,N,K,M] complex
    v_re = tf.transpose(v_re_in, [0, 2, 1, 3])   # [B,N,K,M] float
    v_im = tf.transpose(v_im_in, [0, 2, 1, 3])   # [B,N,K,M] float
    s_bnk = tf.transpose(s, [0, 2, 1])           # [B,N,K]

    # Real-valued equivalent of h^H v to keep downstream math in float.
    h_re = tf.math.real(h_bnkm)
    h_im = tf.math.imag(h_bnkm)
    # Re{h^H v} = sum_m (h_re v_re + h_im v_im)
    # Im{h^H v} = sum_m (h_re v_im - h_im v_re)
    re_proj = tf.einsum("bnkm,bnum->bnku", h_re, v_re) + tf.einsum("bnkm,bnum->bnku", h_im, v_im)
    im_proj = tf.einsum("bnkm,bnum->bnku", h_re, v_im) - tf.einsum("bnkm,bnum->bnku", h_im, v_re)
    gain = re_proj * re_proj + im_proj * im_proj

    desired = tf.linalg.diag_part(gain) * s_bnk
    total_rx = tf.einsum("bnu,bnku->bnk", s_bnk, gain)
    interf = tf.maximum(total_rx - desired, 0.0)

    sigma2_t = tf.convert_to_tensor(sigma2, dtype=desired.dtype)
    eps_t = tf.convert_to_tensor(eps, dtype=desired.dtype)
    sinr_bnk = desired / (interf + sigma2_t + eps_t)
    rate_bnk = tf.cast(B_RB, sinr_bnk.dtype) * tf.math.log(1.0 + sinr_bnk) / tf.math.log(2.0)

    sinr = tf.transpose(sinr_bnk, [0, 2, 1])
    rate_kn = tf.transpose(rate_bnk, [0, 2, 1])
    user_rate = tf.reduce_sum(rate_kn, axis=-1)
    p_tot = tf.reduce_sum(s, axis=[1, 2])
    return sinr, rate_kn, user_rate, p_tot


def compute_mtl_losses(
    h_true: tf.Tensor,
    outputs: dict[str, tf.Tensor],
    sigma2: float,
    B_RB: float,
    P_RB_max: float,
    P_tot: float,
    U_max: int,
    R_max: int,
    train_cfg: MTLTrainConfig,
    rate_target_bps: float = 0.0,
) -> dict[str, tf.Tensor]:
    """Build unsupervised MTL losses (schedule + beam + power)."""
    q = outputs["q"]         # [B,K,N_RB]
    q_all = outputs["q_all"] # [B,K,N_RB+1]
    v_re = outputs["v_re"]   # [B,K,N_RB,M]
    v_im = outputs["v_im"]   # [B,K,N_RB,M]
    p = outputs["p"]         # [B,K,N_RB]
    s = outputs["s"]         # [B,K,N_RB]

    eps = train_cfg.eps

    sinr, rate_kn, user_rate, p_tot_vec = _sinr_and_rates(
        h_true=h_true,
        v_re_in=v_re,
        v_im_in=v_im,
        s=s,
        sigma2=sigma2,
        B_RB=B_RB,
        eps=eps,
    )

    # ---------- Task 1: Scheduling loss ----------
    assign_per_user = tf.reduce_sum(q, axis=-1)                         # [B,K]
    load_per_rb = tf.reduce_sum(q, axis=1)                              # [B,N_RB]

    # R_max upper bound and "at least one RB" lower bound (soft).
    c_user = tf.nn.relu(assign_per_user - float(R_max))
    c_user_low = tf.nn.relu(1.0 - assign_per_user)
    c_rb = tf.nn.relu(load_per_rb - float(U_max))

    entropy = -tf.reduce_sum(q_all * tf.math.log(q_all + eps), axis=-1) # [B,K]
    unscheduled_prob = q_all[..., -1]                                    # [B,K]

    loss_sched = tf.reduce_mean(c_user ** 2) + tf.reduce_mean(c_user_low ** 2) + tf.reduce_mean(c_rb ** 2)
    loss_sched += train_cfg.entropy_weight * tf.reduce_mean(entropy)
    loss_sched += train_cfg.unscheduled_weight * tf.reduce_mean(unscheduled_prob)

    # ---------- Task 2: Beamforming loss ----------
    # Recompute desired/interference components from real-valued h^H v.
    h_re = tf.math.real(h_true)
    h_im = tf.math.imag(h_true)
    re_hv = tf.reduce_sum(h_re * v_re + h_im * v_im, axis=-1)
    im_hv = tf.reduce_sum(h_re * v_im - h_im * v_re, axis=-1)
    desired = s * (re_hv * re_hv + im_hv * im_hv)

    # Approximate total received power from sinr formula:
    # sinr = desired/(interf+sigma2) => interf ≈ desired/sinr - sigma2
    interf = tf.maximum(desired / (sinr + eps) - sigma2, 0.0)

    loss_beam = -tf.reduce_mean(tf.math.log(desired + eps))
    loss_beam += train_cfg.leakage_weight * tf.reduce_mean(interf)

    # ---------- Task 3: Power/QoS loss ----------
    p_rb = tf.reduce_sum(p, axis=1)                                      # [B,N_RB]
    pen_rb = tf.reduce_mean(tf.nn.relu(p_rb - P_RB_max) ** 2)
    pen_tot = tf.reduce_mean(tf.nn.relu(p_tot_vec - P_tot) ** 2)

    # QoS shortfall by user rate target (optional)
    qos_short = tf.nn.relu(rate_target_bps - user_rate)

    loss_power = tf.reduce_mean(p_tot_vec / (P_tot + eps))
    loss_power += pen_rb + pen_tot
    loss_power += train_cfg.qos_weight * tf.reduce_mean(qos_short)

    # ---------- Weighted MTL sum ----------
    total_loss = (
        train_cfg.w_sched * loss_sched
        + train_cfg.w_beam * loss_beam
        + train_cfg.w_power * loss_power
    )

    return {
        "loss_total": total_loss,
        "loss_sched": loss_sched,
        "loss_beam": loss_beam,
        "loss_power": loss_power,
        "mean_sum_rate": tf.reduce_mean(tf.reduce_sum(rate_kn, axis=[1, 2])),
        "mean_user_rate": tf.reduce_mean(user_rate),
        "qos_user_ok_frac": tf.reduce_mean(tf.cast(user_rate >= rate_target_bps, tf.float32))
        if rate_target_bps > 0.0
        else tf.constant(0.0, dtype=tf.float32),
        "mean_p_tot": tf.reduce_mean(p_tot_vec),
        "mean_assign": tf.reduce_mean(assign_per_user),
        "mean_load": tf.reduce_mean(load_per_rb),
    }


def train_step(
    model: MTLUnsupervisedModel,
    optimizer: tf.keras.optimizers.Optimizer,
    h_true: tf.Tensor,
    h_hat: tf.Tensor,
    sigma2: float,
    B_RB: float,
    R_max: int,
    train_cfg: MTLTrainConfig,
    rate_target_bps: float = 0.0,
) -> dict[str, tf.Tensor]:
    """One unsupervised training step."""
    with tf.GradientTape() as tape:
        outputs = model(h_hat, training=True)
        losses = compute_mtl_losses(
            h_true=h_true,
            outputs=outputs,
            sigma2=sigma2,
            B_RB=B_RB,
            P_RB_max=model.P_RB_max,
            P_tot=model.P_tot,
            U_max=model.U_max,
            R_max=R_max,
            train_cfg=train_cfg,
            rate_target_bps=rate_target_bps,
        )

    grads = tape.gradient(losses["loss_total"], model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return losses


__all__ = [
    "MTLTrainConfig",
    "MTLUnsupervisedModel",
    "compute_mtl_losses",
    "train_step",
]
