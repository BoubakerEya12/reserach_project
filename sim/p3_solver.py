# sim/p3_solver.py
import numpy as np
from .sinr_balance import sinr_from_beams  # si tu as déjà une fonction SINR
from .scheduler import top_u_users         # même scheduler que P1/P2
from .channels import effective_channel    # ou ce que tu utilises déjà

def wsr_wmmse_rb(H_eff, sigma2, alpha, P_rb, max_iter=50, tol=1e-4):
    """
    Classical WMMSE for weighted sum rate on ONE RB.

    H_eff : [U, M] complex  (effective channels sqrt(beta_u) h_u)
    sigma2: float, noise power on the RB
    alpha : [U] weights (for P3, e.g. alpha_u = 1, or 1/log(1+beta_u)...)
    P_rb  : float, max power allowed on this RB

    Returns:
        w : [U, M] complex beam matrix (row u = w_u)
        p : [U] real powers, ||w_u||^2 = p_u
        sr: scalar, weighted sum rate on this RB
    """
    U, M = H_eff.shape
    # Initialisation: ZF equal power (tu peux réutiliser ta fonction zf_beams)
    V = zf_beams(H_eff)                # [U, M]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    p = np.full(U, P_rb / U)
    W = (np.sqrt(p)[:, None]) * V      # beams

    for _ in range(max_iter):
        # 1) Receiver filters g_u
        Hv = H_eff @ W.T               # [U, U] effective channel
        signal = np.diag(Hv)
        interf = np.sum(np.abs(Hv)**2, axis=1) - np.abs(signal)**2
        g = signal / (interf + sigma2)     # [U]

        # 2) MSE and weights
        mse = 1 - 2*np.real(g*signal) + (np.abs(g)**2)*(interf + sigma2)
        w_mse = alpha / mse            # [U]

        # 3) Update precoder (closed form)
        # A = sum_u alpha_u |g_u|^2 h_u h_u^H  + lambda I
        A = np.zeros((M, M), dtype=np.complex128)
        B = np.zeros((M, U), dtype=np.complex128)
        for u in range(U):
            hu = H_eff[u][:, None]     # [M,1]
            A += w_mse[u] * (np.abs(g[u])**2) * (hu @ hu.conj().T)
            B[:, u] = w_mse[u] * g[u].conj() * H_eff[u].conj()

        # waterfilling-like scaling via lambda chosen for power P_rb
        # W_mmse = A^{-1} B
        W_new = np.linalg.solve(A, B)  # [M, U]
        W_new = W_new.T                # [U, M]
        # enforce total power constraint
        pow_tot = np.sum(np.linalg.norm(W_new, axis=1)**2)
        if pow_tot > 0:
            W_new *= np.sqrt(P_rb / pow_tot)

        # convergence check
        if np.max(np.abs(W_new - W)) < tol:
            W = W_new
            break
        W = W_new

    # final powers + WSR
    p = np.linalg.norm(W, axis=1)**2
    Hv = H_eff @ W.T
    signal = np.diag(Hv)
    interf = np.sum(np.abs(Hv)**2, axis=1) - np.abs(signal)**2
    sinr = np.abs(signal)**2 / (interf + sigma2)
    rates = np.log2(1 + sinr)
    wsr = float(np.sum(alpha * rates))
    return W, p, wsr, sinr
