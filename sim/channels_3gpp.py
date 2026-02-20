import numpy as np
from typing import Tuple

# =========================================================
# 3GPP TR 38.901 Urban Macro (UMa): path loss + shadowing
#
# PURPOSE
#   Produce PER-USER large-scale gains beta_u in LINEAR scale.
#   This large-scale gain multiplies the small-scale fading vector h_{u,n}.
#
# KEY DEFINITIONS
#   - Path loss (PL_dB): positive number in dB (e.g., 110 dB). Larger = worse link.
#   - Linear gain (beta): what we use in the baseband model. Convert via:
#         beta = 10^(- PL_dB / 10)
#     Example: PL_dB = 100 dB -> beta = 10^(-10) = 1e-10 (very small).
# =========================================================


def los_probability_uma(d2d_m: np.ndarray) -> np.ndarray:
    """
    Compute the probability of Line-of-Sight (LOS) for UMa.

    INPUTS
      d2d_m: np.ndarray of shape [K]
             Horizontal distances (meters) from BS to each user.

    OUTPUTS
      P_LOS: np.ndarray of shape [K], float32, each in [0, 1]

    MODEL (3GPP-like piecewise function)
      If d <= 18 m: P_LOS = 1
      If d > 18 m : P_LOS = (18/d)*(1 - exp(-(d-18)/63)) + exp(-(d-18)/63)
    """
    # Guard against division by zero.
    d = np.maximum(d2d_m, 1e-9)

    # Exponential term from the 3GPP expression.
    Pexp = np.exp(-(d - 18.0) / 63.0)

    # Combine per the formula; this smoothly decays with distance.
    P = (18.0 / d) * (1.0 - Pexp) + Pexp

    # Clip short distances to probability 1.
    P = np.where(d <= 18.0, 1.0, P)

    # Ensure numeric stability and type.
    return np.clip(P, 0.0, 1.0).astype(np.float32)


def pl_uma_los_db(d3d_m: np.ndarray, fc_GHz: float) -> np.ndarray:
    """
    LOS path loss (in dB) for UMa using a standard simplified 3GPP form.

    FORMULA
      PL_LOS(dB) = 28 + 22*log10(d_3D) + 20*log10(fc_GHz)

    INPUTS
      d3d_m : [K] 3D distances (meters)
      fc_GHz: scalar carrier frequency in GHz

    RETURNS
      PL_LOS_dB: [K] LOS path loss in dB (positive values)
    """
    # Avoid log10(0) at extremely small distances.
    d3d_m = np.maximum(d3d_m, 1.0)

    return (28.0
            + 22.0 * np.log10(d3d_m)
            + 20.0 * np.log10(fc_GHz)).astype(np.float32)


def pl_uma_nlos_db(d3d_m: np.ndarray, fc_GHz: float, h_ut: float) -> np.ndarray:
    """
    NLOS 'prime' path loss (in dB) for UMa (3GPP form).

    FORMULA
      PL'_NLOS(dB) = 13.54 + 39.08*log10(d_3D)
                     + 20*log10(fc_GHz) - 0.6*(h_UT - 1.5)

    NOTE
      3GPP then sets: PL_NLOS = max(PL_LOS, PL'_NLOS)
      (NLOS cannot be better than LOS.)

    INPUTS
      d3d_m : [K] 3D distances (meters)
      fc_GHz: scalar frequency in GHz
      h_ut  : user terminal height (meters)

    RETURNS
      PL_NLOS_prime_dB: [K] NLOS' path loss in dB
    """
    d3d_m = np.maximum(d3d_m, 1.0)

    return (13.54
            + 39.08 * np.log10(d3d_m)
            + 20.0  * np.log10(fc_GHz)
            - 0.6   * (h_ut - 1.5)).astype(np.float32)

def sample_user_distances(K, cell_radius_m, h_bs, h_ut, rng):
    """
    Sample K users uniformly over a disk (annulus) and compute distances.
    Robustly forces scalar heights and radius.
    """
    # --- Force scalar types ---
    cell_radius_m = float(np.mean(np.atleast_1d(cell_radius_m)))
    h_bs = float(np.mean(np.atleast_1d(h_bs)))
    h_ut = float(np.mean(np.atleast_1d(h_ut)))

    # --- Radial sampling (area-uniform) ---
    r = np.sqrt(rng.uniform(10.0**2, cell_radius_m**2, size=K)).astype(np.float32)

    # --- Vertical offset (scalar) ---
    dz = abs(h_bs - h_ut)

    # --- 3-D distance ---
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
    Generate per-user large-scale gains beta_u (LINEAR) with UMa LOS/NLOS + shadowing.

    PIPELINE
      1) Place K users -> get d2d (horizontal) and d3d (3D) distances
      2) Decide LOS/NLOS (probabilistic or forced)
      3) Compute path-loss in dB (LOS or max(LOS, NLOS'))
      4) Add log-normal shadowing in dB (Gaussian in dB domain)
      5) Convert dB loss to LINEAR gain: beta = 10^(-PL_dB/10)

    INPUTS
      K, fc_GHz, h_bs, h_ut, los_mode, cell_radius_m, rng

    RETURNS
      beta   : [K] linear gains (path loss + shadowing)
      is_los : [K] booleans (True if LOS, else NLOS)
      d2d_m  : [K] horizontal distances (m)
    """
    # 1) User placement (distances)
    d2d_m, d3d_m = sample_user_distances(K, cell_radius_m, h_bs, h_ut, rng)

    # 2) LOS/NLOS decision
    if los_mode == "always_los":
        is_los = np.ones(K, dtype=bool)
    elif los_mode == "always_nlos":
        is_los = np.zeros(K, dtype=bool)
    else:
        p_los = los_probability_uma(d2d_m)
        # Bernoulli draw per user using P_LOS
        is_los = rng.uniform(size=K) < p_los

    # 3) Path-loss in dB
    pl_los = pl_uma_los_db(d3d_m, fc_GHz)              # LOS branch
    pl_nlos_prime = pl_uma_nlos_db(d3d_m, fc_GHz, h_ut) # NLOS' branch
    pl_nlos = np.maximum(pl_los, pl_nlos_prime)         # enforce NLOS >= LOS
    PL_dB = np.where(is_los, pl_los, pl_nlos).astype(np.float32)

    # 4) Shadowing in dB (zero-mean Gaussian in dB domain)
    #    Typical std: 4 dB for LOS, 6 dB for NLOS in UMa.
    sf = np.empty(K, dtype=np.float32)
    sf[ is_los] = rng.normal(0.0, 4.0,  size=is_los.sum()).astype(np.float32)
    sf[~is_los] = rng.normal(0.0, 6.0,  size=(~is_los).sum()).astype(np.float32)
    PL_dB = PL_dB + sf   # still in dB

    # 5) Convert to LINEAR gain for baseband usage
    #    This matches my7 system: beta_u = 10^{-(PL_dB)/10}
    beta = 10.0 ** (-PL_dB / 10.0)

    return beta.astype(np.float32), is_los, d2d_m


# ----------------------------------------------------------------
# Wrapper for legacy function name "log_distance_shadowing(...)"
# Some older code expects this function; we map to UMa defaults.
# ----------------------------------------------------------------
def log_distance_shadowing(rng, K, fc_GHz):
    """
    Compatibility wrapper: produce (beta, d2d) using UMa defaults.
    """
    beta, is_los, d2d_m = gen_large_scale_3gpp_uma(
        K=K,
        fc_GHz=fc_GHz,
        h_bs=25.0,          # typical UMa BS height (meters)
        h_ut=1.5,           # typical UE height (meters)
        los_mode="prob",    # probabilistic LOS/NLOS per 3GPP
        cell_radius_m=250.0,
        rng=rng
    )
    return beta.astype(np.float32), d2d_m.astype(np.float32)
