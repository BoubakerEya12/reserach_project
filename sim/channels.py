# sim/channels.py
import numpy as np
from typing import Tuple

# =========================================================
# Small-scale Rayleigh fading + Effective channel + CSIT model
# =========================================================

def draw_small_scale(K: int, M: int, rng: np.random.Generator) -> np.ndarray:
    r"""
    Convenience: draw one small-scale channel matrix H ~ CN(0, I) of shape [K, M]
    (rows = users, columns = BS antennas).
    """
    re = rng.standard_normal(size=(K, M)).astype(np.float32)
    im = rng.standard_normal(size=(K, M)).astype(np.float32)
    H = (re + 1j * im) / np.sqrt(2.0)
    return H.astype(np.complex64)

def gen_small_scale_iid_rayleigh(M: int, K: int, N_RB: int,
                                 rng: np.random.Generator) -> np.ndarray:
    r"""
    Draw h_{u,n} ~ CN(0, I_M), independent across (u, n, antenna).
    Returns:
        h: [K, N_RB, M], complex64
    """
    re = rng.standard_normal(size=(K, N_RB, M)).astype(np.float32)
    im = rng.standard_normal(size=(K, N_RB, M)).astype(np.float32)
    h = (re + 1j * im) / np.sqrt(2.0)
    return h.astype(np.complex64)

def apply_large_scale(h: np.ndarray, beta: np.ndarray) -> np.ndarray:
    r"""
    Effective channels: \tilde h_{u,n} = sqrt(beta_u) * h_{u,n}.
    Args:
        h:    [K, N_RB, M], complex64
        beta: [K], linear large-scale gain
    Returns:
        h_eff: [K, N_RB, M], complex64
    """
    K, N_RB, M = h.shape
    beta_b = beta.reshape(K, 1, 1).astype(np.float32)  # broadcast over RBs, antennas
    h_eff = np.sqrt(beta_b) * h
    return h_eff.astype(np.complex64)

def apply_csit_model(h_eff: np.ndarray, eta: float,
                     rng: np.random.Generator) -> np.ndarray:
    r"""
    CSIT model: \hat{\tilde h} = eta * \tilde h + sqrt(1 - eta^2) * e,
    where e ~ CN(0, I).
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be in [0, 1].")
    K, N_RB, M = h_eff.shape
    re = rng.standard_normal(size=(K, N_RB, M)).astype(np.float32)
    im = rng.standard_normal(size=(K, N_RB, M)).astype(np.float32)
    e = (re + 1j * im) / np.sqrt(2.0)
    scale = np.sqrt(max(0.0, 1.0 - eta**2))
    h_hat = (eta * h_eff) + (scale * e.astype(np.complex64))
    return h_hat.astype(np.complex64)

def noise_power_per_rb(N0_W_per_Hz: float, B_RB_Hz: float) -> np.float32:
    """AWGN per-RB noise: sigma^2 = N0 * B_RB (Watts)."""
    return np.float32(N0_W_per_Hz * B_RB_Hz)

def make_channels(M: int, K: int, N_RB: int,
                  beta: np.ndarray, eta: float,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Convenience wrapper to generate one slot:
      - small-scale h ~ CN(0,I)
      - effective channel h_eff
      - CSIT estimate h_eff_hat
    Returns:
      h_eff:     [K, N_RB, M], complex64 (truth at receivers)
      h_eff_hat: [K, N_RB, M], complex64 (BS-side for design)
    """
    h = gen_small_scale_iid_rayleigh(M, K, N_RB, rng)
    h_eff = apply_large_scale(h, beta)
    h_eff_hat = apply_csit_model(h_eff, eta, rng)
    return h_eff, h_eff_hat
