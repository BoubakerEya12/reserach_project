import numpy as np
from typing import Tuple

# =========================================================
# 3GPP TR 38.901 Urban Macro (UMa) path loss + shadowing model
# Output: large-scale channel power gains beta_u (in LINEAR scale)
#
# Reminder:
#   - We first compute the path loss in dB: PL_dB (large-scale attenuation)
#   - Then convert it to a linear gain: beta = 10^(-PL_dB/10)
#     → beta ∈ (0, 1], typically very small (~1e-9 … 1e-13)
#   - This beta multiplies the small-scale fading in the effective channel.
# =========================================================


def los_probability_uma(d2d_m: np.ndarray) -> np.ndarray:
    """
    Line-of-Sight (LOS) probability for Urban Macro (UMa) scenario,
    following the simplified 3GPP TR 38.901 model.

    For horizontal distance d (meters):
        - if d <= 18 m:  P_LOS = 1
        - if d > 18 m:   P_LOS = (18/d)*(1 - exp(-(d-18)/63)) + exp(-(d-18)/63)

    Args:
        d2d_m: 1D array [K] of horizontal distances (meters)

    Returns:
        P_LOS: 1D array [K] of LOS probabilities in [0, 1]
    """
    # Avoid division by zero
    d = np.maximum(d2d_m, 1e-9)

    # Exponential decay term
    Pexp = np.exp(-(d - 18.0) / 63.0)

    # Piecewise formula
    P = (18.0 / d) * (1.0 - Pexp) + Pexp
    P = np.where(d <= 18.0, 1.0, P)

    # Clip and cast to float32
    return np.clip(P, 0.0, 1.0).astype(np.float32)


def pl_uma_los_db(d3d_m: np.ndarray, fc_GHz: float) -> np.ndarray:
    """
    UMa LOS (Line-of-Sight) path loss in dB (simplified version).

        PL_LOS(dB) = 28 + 22*log10(d_3D) + 20*log10(fc_GHz)

    where:
        d_3D : 3D distance between BS and UE (in meters)
        fc_GHz : carrier frequency in GHz

    Note:
        This represents a positive path loss value in dB, not a gain.
        Larger PL_LOS → weaker received power.
    """
    d3d_m = np.maximum(d3d_m, 1.0)  # prevent log10(0)
    return (28.0
            + 22.0 * np.log10(d3d_m)
            + 20.0 * np.log10(fc_GHz)).astype(np.float32)


def pl_uma_nlos_db(d3d_m: np.ndarray, fc_GHz: float, h_ut: float) -> np.ndarray:
    """
    UMa NLOS ("prime") path loss in dB according to 3GPP TR 38.901.

        PL'_NLOS(dB) = 13.54 + 39.08*log10(d_3D)
                       + 20*log10(fc_GHz) - 0.6*(h_UT - 1.5)

    3GPP defines:
        PL_NLOS = max(PL_LOS, PL'_NLOS)
    meaning that the NLOS path loss can never be smaller (better) than the LOS one.

    Args:
        d3d_m : 3D distances (meters)
        fc_GHz : carrier frequency (GHz)
        h_ut : UE height (meters)
    """
    d3d_m = np.maximum(d3d_m, 1.0)
    return (13.54
            + 39.08 * np.log10(d3d_m)
            + 20.0  * np.log10(fc_GHz)
            - 0.6   * (h_ut - 1.5)).astype(np.float32)


def sample_user_distances(K: int,
                          cell_radius_m: float,
                          h_bs: float,
                          h_ut: float,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample K user positions within a circular cell area (annulus),
    and compute both 2D and 3D distances.

    Why sqrt(U)?
        If radius r ~ Uniform[0, R], users would cluster near the center.
        For uniform density over the *area*, use r = sqrt(U)*R.

    Returns:
        d2d_m: horizontal distances [K]
        d3d_m: 3D distances including height difference [K]
    """
    r = np.sqrt(rng.uniform(10.0**2, cell_radius_m**2, size=K)).astype(np.float32)
    dz = abs(h_bs - h_ut)
    d3d = np.sqrt(r**2 + dz**2).astype(np.float32)
    return r, d3d


def gen_large_scale_3gpp_uma(K: int,
                             fc_GHz: float,
                             h_bs: float,
                             h_ut: float,
                             los_mode: str,
                             cell_radius_m: float,
                             rng: np.random.Generator
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate large-scale channel gains (beta_u) for a 3GPP Urban Macro (UMa) scenario.

    Steps:
      1) Sample user distances (2D and 3D)
      2) Decide LOS/NLOS state (probabilistic or forced)
      3) Compute path loss PL_LOS or PL_NLOS (dB)
      4) Add log-normal shadowing (in dB)
      5) Convert path loss to linear gain: beta = 10^(-PL_dB/10)

    Args:
        K : number of users
        fc_GHz : carrier frequency (GHz)
        h_bs, h_ut : BS and UE heights (m)
        los_mode : "prob" | "always_los" | "always_nlos"
        cell_radius_m : cell radius (m)
        rng : numpy random generator for reproducibility

    Returns:
        beta   : [K] linear large-scale power gains (path loss × shadowing)
        is_los : [K] boolean LOS indicators
        d2d_m  : [K] horizontal distances
    """
    # 1) User placement
    d2d_m, d3d_m = sample_user_distances(K, cell_radius_m, h_bs, h_ut, rng)

    # 2) LOS/NLOS decision
    if los_mode == "always_los":
        is_los = np.ones(K, dtype=bool)
    elif los_mode == "always_nlos":
        is_los = np.zeros(K, dtype=bool)
    else:
        p_los = los_probability_uma(d2d_m)
        is_los = rng.uniform(size=K) < p_los  # Bernoulli draw

    # 3) Path loss (dB)
    pl_los = pl_uma_los_db(d3d_m, fc_GHz)
    pl_nlos_prime = pl_uma_nlos_db(d3d_m, fc_GHz, h_ut)
    pl_nlos = np.maximum(pl_los, pl_nlos_prime)  # enforce PL_NLOS ≥ PL_LOS 
    PL_dB = np.where(is_los, pl_los, pl_nlos).astype(np.float32) 

    # 4) Shadow fading (σ = 4 dB for LOS, 6 dB for NLOS)
    sf = np.empty(K, dtype=np.float32)
    sf[ is_los] = rng.normal(0.0, 4.0,  size=is_los.sum()).astype(np.float32)
    sf[~is_los] = rng.normal(0.0, 6.0,  size=(~is_los).sum()).astype(np.float32)
    PL_dB = PL_dB + sf  # still in dB

    # 5) Convert to linear gain (same as Eq. (1) in your LaTeX system model)
    #    Example: PL_dB = 100 → beta = 10^(-10) = 1e-10
    beta = 10.0 ** (-PL_dB / 10.0)

    return beta.astype(np.float32), is_los, d2d_m


# --- Compatibility wrapper for older scripts ---
def log_distance_shadowing(rng, K, fc_GHz):
    """
    Wrapper for backward compatibility.
    Some older scripts call log_distance_shadowing(...),
    so we simply redirect to the 3GPP UMa generator with default parameters.
    """
    beta, is_los, d2d_m = gen_large_scale_3gpp_uma(
        K=K,
        fc_GHz=fc_GHz,
        h_bs=25.0,          # Typical BS height for UMa
        h_ut=1.5,           # Typical UE height
        los_mode="prob",    # Probabilistic LOS/NLOS
        cell_radius_m=250.0,
        rng=rng
    )
    return beta.astype(np.float32), d2d_m.astype(np.float32)
