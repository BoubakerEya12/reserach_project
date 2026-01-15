# sim/scheduler.py
import numpy as np
from typing import Dict, Literal

def pick_users_per_rb(h_hat_rb: np.ndarray, U_max: int) -> np.ndarray:
    scores = np.sum(np.abs(h_hat_rb)**2, axis=1)   # [K]
    order = np.argsort(-scores)                    # descending
    U = min(U_max, int(np.sum(scores > 0)))
    return order[:U]

def zf_precoder(H: np.ndarray) -> np.ndarray:
    """
    W = H (H^H H)^(-1), H: [M,U] (columns = user channels).
    """
    HH = H.conj().T @ H
    eps = 1e-6 * (np.trace(HH).real / max(HH.shape[0], 1))
    HH_inv = np.linalg.pinv(HH + eps * np.eye(HH.shape[0], dtype=HH.dtype))
    return H @ HH_inv

def rzf_precoder(H: np.ndarray, alpha: float) -> np.ndarray:
    """
    Regularized ZF (a.k.a. MMSE precoder):
      W = H (H^H H + alpha I_U)^(-1)
    """
    U = H.shape[1]
    HH = H.conj().T @ H
    A = HH + alpha * np.eye(U, dtype=HH.dtype)
    A_inv = np.linalg.pinv(A)
    return H @ A_inv

def normalize_columns(W: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W, axis=0) + 1e-12
    return (W / norms).astype(np.complex64)

def schedule_and_power(
    h_eff: np.ndarray,
    h_eff_hat: np.ndarray,
    sigma2: float,
    P_RB_max: float,
    P_tot: float,
    U_max: int,
    beamformer: Literal["zf", "rzf"] = "rzf",
) -> Dict[str, np.ndarray]:
    """
    Baseline greedy + ZF/RZF (for comparison).
    Args:
        h_eff:     [K, N_RB, M] TRUE channels (for SINR)
        h_eff_hat: [K, N_RB, M] ESTIMATED channels (for beam design)
        sigma2:    AWGN power per RB (W)
        P_RB_max:  per-RB power cap (W)
        P_tot:     total slot power (W)
        U_max:     max users per RB
        beamformer:"zf" or "rzf"
    Returns:
        assign_rb : [K] int RB index or -1
        beams     : [N_RB, M, K] complex64, v_{u,n} (unit-norm; zeros if unscheduled)
        powers    : [N_RB, K] float32, p_{u,n} (W; 0 if unscheduled)
        sinr      : [N_RB, K] float32
        alpha_used: [N_RB] float32, alpha per RB actually used (0 for ZF; U*sigma2/P_rb for RZF)
    """
    K, N_RB, M = h_eff.shape
    assign_rb  = -np.ones(K, dtype=np.int32)
    beams      = np.zeros((N_RB, M, K), dtype=np.complex64)
    powers     = np.zeros((N_RB, K), dtype=np.float32)
    sinr       = np.zeros((N_RB, K), dtype=np.float32)
    alpha_used = np.zeros((N_RB,),     dtype=np.float32)

    remaining_P = float(P_tot)

    for n in range(N_RB):
        # Select users by strength (estimated channels)
        hhat_rb = h_eff_hat[:, n, :]   # [K,M]
        sel = pick_users_per_rb(hhat_rb, U_max)
        U = sel.size
        if U == 0 or remaining_P <= 0.0:
            continue

        P_rb = min(P_RB_max, remaining_P)
        if P_rb <= 0.0:
            continue

        if beamformer == "rzf":
            alpha_n = float(U * sigma2 / max(P_rb, 1e-12))
        else:
            alpha_n = 0.0
        alpha_used[n] = np.float32(alpha_n)

        # Build precoder on estimates for selected users
        H = hhat_rb[sel, :].T  # [M,U]
        if alpha_n > 0.0:
            W = rzf_precoder(H, alpha=alpha_n)     # [M,U]
        else:
            W = zf_precoder(H)                     # [M,U]
        V = normalize_columns(W)                   # [M,U]

        # Equal power among U users on this RB
        p_each = np.float32(P_rb / U)

        for j, u in enumerate(sel):
            beams[n, :, u] = V[:, j]
            powers[n, u]   = p_each
            assign_rb[u]   = n

        remaining_P -= P_rb
        if remaining_P < 0:
            remaining_P = 0.0

        # SINR with TRUE channels
        htrue_rb = h_eff[:, n, :]   # [K,M]
        for j, u in enumerate(sel):
            v_u = V[:, j]
            sig = p_each * np.abs(htrue_rb[u, :].conj() @ v_u)**2
            interf = 0.0
            for k, v in enumerate(sel):
                if v == u:
                    continue
                interf += p_each * np.abs(htrue_rb[u, :].conj() @ V[:, k])**2
            sinr[n, u] = np.float32(sig / (interf + sigma2))

    return dict(
        assign_rb=assign_rb,
        beams=beams,
        powers=powers,
        sinr=sinr,
        alpha_used=alpha_used,
    )
