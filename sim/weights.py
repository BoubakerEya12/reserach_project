# sim/weights.py
import numpy as np

def compute_alpha_from_beta(beta: np.ndarray,
                            gamma: float = 0.5,
                            eps: float = 1e-15) -> np.ndarray:
    """
    Compute user weights α_u for P3 from large-scale gains β_u.

    INPUT
    -----
    beta : np.ndarray of shape [K]
        Large-scale gains (path loss + shadowing) in LINEAR scale.
        Larger beta_u = better link.

    PARAMETERS
    ----------
    gamma : float
        Exponent controlling how strongly we favor strong users.
        - gamma = 0   -> all α_u equal (P3 ~= P2/P1 style)
        - 0 < gamma < 1 -> soft preference for strong users
        - gamma = 1   -> α_u ∝ beta_u (plus agressif)
    eps : float
        Small floor to avoid zeros.

    OUTPUT
    ------
    alpha : np.ndarray of shape [K]
        User weights α_u, non-negative, with:
            sum_u α_u = K   (i.e. moyenne = 1)

    PROPERTIES
    ----------
    - α_u is strictly increasing in β_u
    - Si un user a un meilleur canal (plus grand beta), il obtient
      un poids plus grand dans la WSR de P3.
    """
    beta = np.asarray(beta, dtype=np.float64)
    K = beta.shape[0]

    # Clip pour éviter beta = 0
    beta_clipped = np.maximum(beta, eps)

    # Poids bruts proportionnels à beta^gamma
    raw = beta_clipped ** gamma   # [K]

    s = raw.sum()
    if s <= 0 or not np.isfinite(s):
        # Cas pathologique : on retombe sur des poids uniformes
        return np.ones(K, dtype=np.float64)

    # Normalisation : moyenne(alpha_u) = 1 -> somme = K
    alpha = raw * (K / s)

    return alpha.astype(np.float64)
