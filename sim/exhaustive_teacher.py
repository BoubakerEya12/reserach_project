import numpy as np
import itertools
from typing import Tuple, Optional

from .sinr_balance import sinr_balancing_power_constraint
from scripts.common import p1_maxmin_sinr_db, p2_min_power, sinr_vector
# ↑ If relative imports are required in your layout, use:
# from ..scripts.common import p1_maxmin_sinr_db, p2_min_power, sinr_vector

def best_group_by_balanced_sinr(
    Hn: np.ndarray,            # [M,K] columns are users
    sigma2: float,
    P_rb: float,
    U_max: int,
    rho: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    M, K = Hn.shape
    U_cap = min(U_max, M, K)
    best_score = -np.inf
    best_sel = None
    best_Wdl = None
    best_p = None
    best_sinr = None

    users = list(range(K))
    for U in range(1, U_cap + 1):
        for comb in itertools.combinations(users, U):
            sel = np.array(comb, dtype=int)
            H_sel = Hn[:, sel]  # [M,U]
            Wdl, _, sinr_dl, _, _ = sinr_balancing_power_constraint(
                H=H_sel, Pmax=P_rb, rho=(np.ones(U) if rho is None else rho[:U]), sigma2=sigma2
            )
            score = float(np.min(sinr_dl))
            if score > best_score:
                best_score = score
                best_sel   = sel
                best_Wdl   = Wdl.copy()
                best_p     = (np.linalg.norm(Wdl, axis=0)**2).astype(np.float32)
                best_sinr  = sinr_dl.astype(np.float32)

    beams = np.zeros((M, K), dtype=np.complex64)
    powers = np.zeros((K,), dtype=np.float32)
    sinr_u = np.zeros((K,), dtype=np.float32)
    if best_sel is not None:
        for j, u in enumerate(best_sel):
            v = best_Wdl[:, j]
            norm = np.linalg.norm(v) + 1e-12
            beams[:, u] = (v / norm).astype(np.complex64)
            powers[u]   = (norm**2).astype(np.float32) if isinstance(norm, np.ndarray) else np.float32(norm**2)
            sinr_u[u]   = best_sinr[j]
    return best_sel, beams, powers, sinr_u, float(best_score)

def exhaustive_p1_over_subsets_tuple(
    HkM: np.ndarray,          # [K,M] (rows = users)
    sigma2: float,
    P_rb: float,
    U_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    K, M = HkM.shape
    users = np.arange(K, dtype=int)

    best_gamma_lin = -np.inf
    best_sel = None
    best_W = None
    best_p = None
    best_sinr = None

    for U in range(1, min(U_max, K) + 1):
        for sel in itertools.combinations(users, U):
            idx = np.array(sel, dtype=int)
            H = HkM[idx, :]
            gamma_db = p1_maxmin_sinr_db(H, sigma2, P_rb)
            gamma_lin = 10.0**(gamma_db/10.0)
            totP, p, W = p2_min_power(H, np.full(U, gamma_lin), sigma2)
            if not np.isfinite(totP) or W is None:
                continue
            sinr = sinr_vector(H, W, sigma2)
            g_lin = float(np.min(sinr))
            if g_lin > best_gamma_lin:
                best_gamma_lin = g_lin
                best_sel  = idx
                best_W    = W.copy()                       # [M,U]
                best_p    = (np.linalg.norm(W, axis=0)**2).astype(np.float32)
                best_sinr = sinr.astype(np.float32)

    if best_sel is None:
        return (np.array([], dtype=int),
                np.zeros((HkM.shape[1], 0), dtype=np.complex64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                -300.0)
    return best_sel, best_W.astype(np.complex64), best_p, best_sinr, 10.0*np.log10(max(best_gamma_lin, 1e-30))

def best_value_by_balanced_sinr(
    HkM: np.ndarray,  # [K, M]
    sigma2: float,
    P_rb: float,
    U_max: int,
) -> float:
    K, M = HkM.shape
    Hn = HkM.T  # [M,K]
    sel, _, _, _, score_lin = best_group_by_balanced_sinr(
        Hn=Hn, sigma2=sigma2, P_rb=P_rb, U_max=U_max
    )
    # IMPORTANT: return linear min-SINR (not dB)
    return float(max(score_lin, 1e-30))
