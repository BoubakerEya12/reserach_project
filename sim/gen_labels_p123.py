# sim/gen_labels_p123.py
# ----------------------------------------------------------
# Unified label generator for:
#   - P1: max–min SINR with subset selection
#   - P2: min power to meet SINR targets
#   - P3: WSR via WMMSE
#
# It uses:
#   - sim/config.py         -> SystemConfig
#   - sim/common.py         -> channels + P1/P2/P3 solvers
#   - sim/channels_3gpp.py  -> 3GPP UMa large-scale model
#
# All system parameters (K, M, N0, B_RB, P_tot, etc.)
# are taken from SystemConfig, NOT hard-coded.
# ----------------------------------------------------------

from typing import List, Dict, Any
import numpy as np

from .config import SystemConfig
from .common import (
    make_rng,
    draw_iid_small_scale,
    effective_channel,
    sinr_vector,
    # P1 helpers:
    exhaustive_p1_over_subsets,
    p1_solve_fixed_users_tuple,
    # P2 solver:
    p2_min_power,
    # P3 solver:
    wmmse_sumrate,
)
from .channels_3gpp import gen_large_scale_3gpp_uma


def generate_single_sample(
    cfg: SystemConfig,
    rng: np.random.Generator,
    sinr_target_dB: float = 5.0,
) -> Dict[str, Any]:
    """
    Generate ONE channel realization + labels for P1, P2, P3.

    Returns:
      sample = {
        "H_eff": H_eff [K,M] complex64,
        "beta":  beta [K] float32,
        "problems": {
           "P1": {...},
           "P2": {...},
           "P3": {...},
        }
      }

    For each problem i in {P1, P2, P3}:
      problems[i] = {
         "RB_assignment": [idx],# list length = N_RB (here 1 RB, so length=1)
         "W_beams":      [W],   # list length = N_RB, W: [M,U_n]
         "powers":       [p],   # list length = N_RB, p: [U_n]
         "SINRs":        [sinr],# list length = N_RB, sinr: [U_n]
         "objective":    scalar # gamma_dB (P1), total power (P2), sum-rate (P3)
      }
    """
    K = cfg.K
    M = cfg.M
    sigma2 = cfg.sigma2
    Pmax = cfg.P_tot
    Umax = cfg.U_max

    # ---------- 1) Small-scale fading ----------
    Hs = draw_iid_small_scale(rng, K, M)  # [K,M], CN(0,1)

    # ---------- 2) Large-scale 3GPP UMa ----------
    beta, is_los, d2d_m = gen_large_scale_3gpp_uma(
        K=K,
        fc_GHz=cfg.fc_GHz,
        h_bs=cfg.h_bs,
        h_ut=cfg.h_ut,
        los_mode=cfg.los_mode,
        cell_radius_m=cfg.cell_radius_m,
        rng=rng,
    )

    # Effective per-user channel including path loss + shadowing
    H_eff = effective_channel(Hs, beta)  # [K,M] complex

    problems: Dict[str, Any] = {}

    # ======================================================
    # P1: Max–min SINR (balanced SINR) with subset selection
    # ======================================================
    try:
        gamma_star_dB, best_subset = exhaustive_p1_over_subsets(
            H_eff, sigma2, Pmax, Umax=Umax
        )
    except Exception:
        gamma_star_dB = -300.0
        best_subset = tuple([])

    if (best_subset is None) or (len(best_subset) == 0):
        problems["P1"] = {
            "RB_assignment": [np.array([], dtype=np.int32)],
            "W_beams": [np.zeros((M, 0), dtype=np.complex64)],
            "powers": [np.zeros((0,), dtype=np.float32)],
            "SINRs": [np.zeros((0,), dtype=np.float32)],
            "objective": float(gamma_star_dB),
        }
    else:
        idx = np.array(best_subset, dtype=np.int32)
        H_sub = H_eff[idx, :]  # [U1, M]

        W_p1, p_p1, sinr_p1, gamma_db = p1_solve_fixed_users_tuple(
            H_sub, sigma2, Pmax
        )

        problems["P1"] = {
            "RB_assignment": [idx],
            "W_beams": [W_p1.astype(np.complex64)],
            "powers": [p_p1.astype(np.float32)],
            "SINRs": [sinr_p1.astype(np.float32)],
            "objective": float(gamma_db),
        }

    # ======================================================
    # P2: Min power to meet target SINR (reuse P1 subset)
    # ======================================================
    if len(problems["P1"]["RB_assignment"][0]) == 0:
        problems["P2"] = {
            "RB_assignment": [np.array([], dtype=np.int32)],
            "W_beams": [np.zeros((M, 0), dtype=np.complex64)],
            "powers": [np.zeros((0,), dtype=np.float32)],
            "SINRs": [np.zeros((0,), dtype=np.float32)],
            "objective": float("inf"),
        }
    else:
        idx = problems["P1"]["RB_assignment"][0]
        H_sub = H_eff[idx, :]  # [U2, M]

        gamma_target_lin = 10.0 ** (sinr_target_dB / 10.0)
        gamma_vec = np.full(H_sub.shape[0], gamma_target_lin, dtype=np.float64)

        total_power, p_p2, W_p2 = p2_min_power(H_sub, gamma_vec, sigma2)

        if (not np.isfinite(total_power)) or (W_p2 is None):
            problems["P2"] = {
                "RB_assignment": [idx],
                "W_beams": [np.zeros((M, 0), dtype=np.complex64)],
                "powers": [np.zeros((0,), dtype=np.float32)],
                "SINRs": [np.zeros((0,), dtype=np.float32)],
                "objective": float("inf"),
            }
        else:
            sinr_p2 = sinr_vector(H_sub, W_p2, sigma2).astype(np.float32)
            p_vec_p2 = (np.linalg.norm(W_p2, axis=0) ** 2).astype(np.float32)
            problems["P2"] = {
                "RB_assignment": [idx],
                "W_beams": [W_p2.astype(np.complex64)],
                "powers": [p_vec_p2],
                "SINRs": [sinr_p2],
                "objective": float(total_power),
            }

    # ======================================================
    # P3: WSR via WMMSE (all users by default)
    # ======================================================
    try:
        Rsum, W_p3 = wmmse_sumrate(H_eff, sigma2, Pmax, iters=50)
        sinr_p3 = sinr_vector(H_eff, W_p3, sigma2).astype(np.float32)
        p_vec_p3 = (np.linalg.norm(W_p3, axis=0) ** 2).astype(np.float32)
        S_p3 = np.arange(K, dtype=np.int32)
        problems["P3"] = {
            "RB_assignment": [S_p3],
            "W_beams": [W_p3.astype(np.complex64)],
            "powers": [p_vec_p3],
            "SINRs": [sinr_p3],
            "objective": float(Rsum),
        }
    except Exception:
        S_p3 = np.arange(K, dtype=np.int32)
        problems["P3"] = {
            "RB_assignment": [S_p3],
            "W_beams": [np.zeros((M, K), dtype=np.complex64)],
            "powers": [np.zeros((K,), dtype=np.float32)],
            "SINRs": [np.zeros((K,), dtype=np.float32)],
            "objective": 0.0,
        }

    sample = {
        "H_eff": H_eff.astype(np.complex64),
        "beta": beta.astype(np.float32),
        "problems": problems,
    }
    return sample


def generate_dataset(
    cfg: SystemConfig,
    N_samples: int = 1000,
    seed: int | None = None,
    sinr_target_dB: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Generate a list of N_samples labels for P1/P2/P3.

    cfg  : SystemConfig
    seed : if None, use cfg.seed; else override
    """
    rng = make_rng(cfg.seed if seed is None else seed)
    samples: List[Dict[str, Any]] = []
    for _ in range(N_samples):
        s = generate_single_sample(cfg, rng, sinr_target_dB=sinr_target_dB)
        samples.append(s)
    return samples


if __name__ == "__main__":
    cfg = SystemConfig()
    data = generate_dataset(cfg, N_samples=10, sinr_target_dB=5.0)
    print(f"Generated {len(data)} samples")
    print("Keys in one sample:", data[0].keys())
    print("P1 objective (gamma_dB):", data[0]["problems"]["P1"]["objective"])
    print("P2 objective (total power):", data[0]["problems"]["P2"]["objective"])
    print("P3 objective (sum-rate):", data[0]["problems"]["P3"]["objective"])
