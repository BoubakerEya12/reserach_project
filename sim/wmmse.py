# sim/wmmse.py
import numpy as np
from typing import Tuple

def wmmse_sumrate(
    H: np.ndarray,         # [M, U], colonnes = h_u
    sigma2: float,
    Pmax: float,
    iters: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    WMMSE classique (downlink MISO), renvoie (W, p, sinr).
    W est [M,U] avec colonnes w_u (déjà scalées), sum ||w_u||^2 <= Pmax.
    """
    M, U = H.shape
    # init : RZF directions + partage de puissance
    Gram = H.conj().T @ H
    eps = 1e-6 * (np.trace(Gram).real / max(U, 1))
    A = Gram + eps * np.eye(U, dtype=Gram.dtype)
    V = H @ np.linalg.pinv(A)  # [M,U]
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    p = np.full(U, Pmax / max(U, 1), dtype=np.float64)

    for _ in range(iters):
        W = V * np.sqrt(p.reshape(1, -1))  # [M,U]

        # MMSE receivers (scalaires) et poids
        g = np.zeros(U, dtype=np.complex128)  # equalizer scalars
        e = np.zeros(U, dtype=np.float64)     # MSE
        u = np.zeros(U, dtype=np.float64)     # weights
        for k in range(U):
            hk = H[:, k]
            sig = hk.conj() @ W[:, k]
            interf = 0.0
            for j in range(U):
                if j == k: 
                    interf += np.abs(hk.conj() @ W[:, j])**2
            interf += sigma2
            g[k] = sig / interf
            e[k] = 1.0 - 2*np.real(g[k]*sig) + np.abs(g[k])**2 * interf
            e[k] = float(max(e[k], 1e-12))
            u[k] = 1.0 / e[k]

        # update des précoders (waterfilling implicite via KKT)
        # fermer la forme (Shi et al. 2011) : W = (sum u_k |g_k|^2 h_k h_k^H + mu I)^(-1) (sum u_k g_k^* h_k)
        A = np.zeros((M, M), dtype=np.complex128)
        b = np.zeros((M,), dtype=np.complex128)
        for k in range(U):
            hk = H[:, k].reshape(-1, 1)
            A += u[k] * (np.abs(g[k])**2) * (hk @ hk.conj().T)
            b += u[k] * np.conj(g[k]) * hk[:, 0]

        # trouver mu >= 0 tel que sum ||w_k||^2 = Pmax
        # on résout (A + mu I) x = b, w_k = x * (direction vers k) via projection linéaire multi-utilisateurs
        # Ici, on approxime en résolvant W commun puis ré-orthonormalisant comme dans la littérature
        # (approche simple : un seul vecteur 'x' puis projection sur chaque utilisateur)
        # Pour rester stable et simple, on fait une recherche binaire sur mu pour respecter Pmax.
        def power_with_mu(mu: float) -> float:
            X = np.linalg.pinv(A + mu * np.eye(M)) @ b
            # répartir X sur U directions (minimisation L2 par projection)
            # direction naïve : aligner sur colonnes V actuelles
            Vdir = V.copy()
            Vdir = Vdir / (np.linalg.norm(Vdir, axis=0, keepdims=True) + 1e-12)
            coeffs = (Vdir.conj().T @ X)  # [U]
            Wnew = Vdir * coeffs.reshape(1, -1)
            return float(np.sum(np.linalg.norm(Wnew, axis=0)**2)), Wnew

        mu_lo, mu_hi = 0.0, 1e6
        for _ in range(30):
            mu_mid = 0.5*(mu_lo + mu_hi)
            pow_mid, _ = power_with_mu(mu_mid)
            if pow_mid > Pmax:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid
        _, W = power_with_mu(mu_hi)

        # renormalise directions et puissances
        p = np.linalg.norm(W, axis=0)**2
        V = W / (np.sqrt(p.reshape(1, -1)) + 1e-12)

    # SINR final
    sinr = np.zeros(U, dtype=np.float64)
    for k in range(U):
        hk = H[:, k]
        sig = np.abs(hk.conj() @ (V[:, k]*np.sqrt(p[k])))**2
        interf = 0.0
        for j in range(U):
            if j == k: 
                continue
            interf += np.abs(hk.conj() @ (V[:, j]*np.sqrt(p[j])))**2
        sinr[k] = sig / (interf + sigma2)

    Wfinal = V * np.sqrt(p.reshape(1, -1))
    return Wfinal.astype(np.complex64), p.astype(np.float32), sinr.astype(np.float32)
