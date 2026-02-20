# sim/p3_wsr_eval.py
import numpy as np
import tensorflow as tf

from config import SystemConfig
from .channels import draw_small_scale
from .channels_3gpp import gen_large_scale_3gpp_uma
from .weights import compute_alpha_from_beta
from .sionna_channel import generate_h_rb_true


# =====================================================================
# Utilitaires
# =====================================================================

def unit_norm_rows(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalise chaque ligne de V à la norme 1."""
    n = np.sqrt(np.maximum((np.abs(V) ** 2).sum(axis=1, keepdims=True), eps))
    return V / n


def compute_sinr_given_beams(H: np.ndarray,
                             V: np.ndarray,
                             p: np.ndarray,
                             sigma2: float) -> np.ndarray:
    """
    SINR pour un RB donné.

    H : [U, M] canal effectif
    V : [U, M] directions de faisceaux (unit-norm par ligne)
    p : [U] puissances par utilisateur (Watts)
    sigma2 : bruit par RB

    Retourne :
        sinr : [U]
    """
    U, M = H.shape
    assert V.shape == (U, M)
    assert p.shape == (U,)

    # G[u, j] = h_u v_j^T
    G = H @ V.T               # [U, U]
    gains = np.abs(G) ** 2    # [U, U]
    gains_p = gains * p[None, :]  # [U, U]

    sig = np.diag(gains_p)                     # [U]
    interf = gains_p.sum(axis=1) - sig         # [U]

    sinr = sig / (interf + sigma2)
    return sinr


# =====================================================================
# ZF classique
# =====================================================================

def zf_beams(H: np.ndarray) -> np.ndarray:
    """
    ZF linéaire pour MU-MISO.

    H : [U, M] canal effectif
    Retourne :
       V : [U, M], chaque ligne est un faisceau unitaire pour un utilisateur.
    """
    H = np.atleast_2d(H)
    U, M = H.shape
    Hh = H.conj().T                # [M, U]
    A = H @ Hh                     # [U, U]
    Vt = Hh @ np.linalg.pinv(A)    # [M, U] (pseudo-inverse par sécurité)
    V = Vt.T                       # [U, M]
    V = unit_norm_rows(V)
    return V.astype(np.complex64)


def sinr_zf_equal_power(H: np.ndarray,
                        P_tot: float,
                        sigma2: float) -> np.ndarray:
    """
    SINR sous ZF avec puissance égale par utilisateur sur un RB.
    """
    U, _ = H.shape
    if U == 0:
        return np.zeros(0, dtype=np.float64)

    V = zf_beams(H)  # [U, M]
    p = np.full(U, P_tot / U, dtype=np.float64)
    return compute_sinr_given_beams(H, V, p, sigma2)


# =====================================================================
# WMMSE pour WSR (un flux par utilisateur, MU-MISO)
# =====================================================================

def wmmse_precoder(H: np.ndarray,
                   alpha: np.ndarray,
                   P_tot: float,
                   sigma2: float,
                   n_iter: int = 30,
                   eps: float = 1e-9):
    """
    WMMSE pour maximiser la Weighted Sum Rate sur un RB donné.

    H     : [U, M] canal effectif
    alpha : [U] poids WSR (ici ceux de P3)
    P_tot : puissance totale max sur ce RB
    sigma2: bruit

    Retourne :
        V : [U, M] directions de faisceaux (unitaires)
        p : [U] puissances par utilisateur
    """
    H = np.asarray(H, dtype=np.complex128)
    alpha = np.asarray(alpha, dtype=np.float64)
    U, M = H.shape

    if U == 0:
        return np.zeros((0, M), dtype=np.complex64), np.zeros(0, dtype=np.float64)

    # --- Initialisation : ZF + puissance égale ---
    V0 = zf_beams(H).astype(np.complex128)  # [U, M]
    p0 = np.full(U, P_tot / U, dtype=np.float64)
    F = V0.T * np.sqrt(p0)[None, :]         # [M, U] colonnes = f_j

    lam_reg = 1e-6  # petite régularisation pour éviter les matrices singulières

    for _ in range(n_iter):
        # ------------------------------------------------
        # Étape 1 : mise à jour des récepteurs u_eq
        # ------------------------------------------------
        hF = H @ F             # [U, U]
        power = np.abs(hF) ** 2
        interf_plus_noise = sigma2 + power.sum(axis=1)  # [U]

        u_eq = np.empty(U, dtype=np.complex128)
        w = np.empty(U, dtype=np.float64)

        for u in range(U):
            signal = hF[u, u]
            denom = interf_plus_noise[u]
            u_eq[u] = signal / (denom + eps)
            # MSE
            e_u = 1.0 - 2.0 * np.real(u_eq[u] * np.conj(signal)) \
                + (np.abs(u_eq[u]) ** 2) * denom
            e_u = float(max(e_u, 1e-9))
            w[u] = alpha[u] / e_u

        # ------------------------------------------------
        # Étape 2 : mise à jour du précoder F via bisection sur λ
        # ------------------------------------------------
        A = np.zeros((M, M), dtype=np.complex128)
        B = np.zeros((M, U), dtype=np.complex128)

        for u in range(U):
            h_u = H[u, :]
            hhH = np.outer(h_u.conj(), h_u)  # [M, M]
            A += w[u] * (np.abs(u_eq[u]) ** 2) * hhH
            B[:, u] = w[u] * np.conj(u_eq[u]) * h_u.conj()

        def compute_F_for_lambda(lam: float) -> np.ndarray:
            """
            Inversion robuste : on ajoute lam_reg sur la diagonale,
            et on tombe sur la pseudo-inverse si la matrice est mal conditionnée.
            """
            mat = A + (lam + lam_reg) * np.eye(M, dtype=np.complex128)
            try:
                return np.linalg.solve(mat, B)  # [M, U]
            except np.linalg.LinAlgError:
                # fallback : pseudo-inverse
                mat_pinv = np.linalg.pinv(mat)
                return mat_pinv @ B

        # Vérifie si λ = 0 respecte déjà la contrainte de puissance
        lam_low = 0.0
        F_try = compute_F_for_lambda(lam_low)
        P = float(np.real(np.sum(np.abs(F_try) ** 2)))
        if P <= P_tot * (1.0 + 1e-3):
            F = F_try
            continue

        # Sinon, cherche un λ_high suffisant
        lam_high = 1.0
        while True:
            F_high = compute_F_for_lambda(lam_high)
            P_high = float(np.real(np.sum(np.abs(F_high) ** 2)))
            if P_high <= P_tot or lam_high > 1e9:
                break
            lam_high *= 2.0

        # Bisection sur λ
        for _ in range(25):
            lam_mid = 0.5 * (lam_low + lam_high)
            F_mid = compute_F_for_lambda(lam_mid)
            P_mid = float(np.real(np.sum(np.abs(F_mid) ** 2)))
            if P_mid > P_tot:
                lam_low = lam_mid
            else:
                lam_high = lam_mid
                F_try = F_mid
        F = F_try

    # ------------------------------------------------
    # Convertit F -> (V, p) avec V lignes unitaires
    # ------------------------------------------------
    p = np.real(np.sum(np.abs(F) ** 2, axis=0))  # [U]
    V = np.zeros((U, M), dtype=np.complex128)
    for u in range(U):
        if p[u] > eps:
            v_col = F[:, u] / np.sqrt(p[u])
            V[u, :] = v_col
        else:
            e = np.zeros(M, dtype=np.complex128)
            e[0] = 1.0
            V[u, :] = e

    V = unit_norm_rows(V)
    return V.astype(np.complex64), p.astype(np.float64)


# =====================================================================
# Sélection des utilisateurs par RB (scheduling)
# =====================================================================

def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Retourne les indices des k plus grands scores."""
    k = int(k)
    k = max(1, k)
    if k >= len(scores):
        return np.arange(len(scores), dtype=int)
    return np.argpartition(scores, -k)[-k:]


# =====================================================================
# Simulation d'un slot pour P3
# =====================================================================

def simulate_one_slot_p3(cfg: SystemConfig,
                         rng: np.random.Generator,
                         mode: str = "zf_beta",
                         sigma2_override: float | None = None,
                         h_rb_override: np.ndarray | None = None) -> float:
    """
    Simule un slot complet pour P3 et retourne la WSR (bits/s/Hz).

    mode ∈ {
       "zf_beta",          # ZF, scheduling sur beta
       "zf_random",        # ZF, utilisateurs aléatoires
       "wmmse_alpha_beta", # WMMSE, scheduling sur alpha*beta
       "mrt_single"        # MRT single-user (upper bound)
    }
    """
    M = cfg.M
    K = cfg.K
    N_RB = cfg.N_RB
    U_max = cfg.U_max
    P_RB_max = cfg.P_RB_max

    # Bruit
    sigma2 = float(cfg.sigma2 if sigma2_override is None else sigma2_override)

    # --- 1) Large-scale + small-scale (Sionna) OR legacy 3GPP+Rayleigh ---
    if h_rb_override is not None:
        # h_rb_override: [K, N_RB, M] already includes large+small scale
        H_rb_all = h_rb_override
        beta_raw = np.mean(np.abs(H_rb_all) ** 2, axis=(1, 2))
        beta_ref = np.median(beta_raw)
        if beta_ref <= 0:
            beta_ref = 1.0
        # Normalize overall scale so median large-scale power ~ 1
        H_rb_all = H_rb_all / np.sqrt(beta_ref)
        beta = beta_raw / beta_ref
    else:
        beta, is_los, d2d_m = gen_large_scale_3gpp_uma(
            K=K,
            fc_GHz=cfg.fc_GHz,
            h_bs=cfg.h_bs,
            h_ut=cfg.h_ut,
            los_mode=cfg.los_mode,
            cell_radius_m=cfg.cell_radius_m,
            rng=rng
        )  # beta : [K]

    # 🔵 NORMALISATION POUR P3 UNIQUEMENT (legacy pathloss case)
    if h_rb_override is None:
        beta_ref = np.median(beta)
        if beta_ref <= 0:
            beta_ref = 1.0
        beta = beta / beta_ref

    # Poids alpha pour P3 (sum alpha_u = K par construction)
    alpha = compute_alpha_from_beta(beta)  # [K]

    WSR_slot = 0.0

    # --- 2) Boucle sur les RB ---
    for n in range(N_RB):
        if h_rb_override is not None:
            H_eff_all = H_rb_all[:, n, :]  # [K, M]
        else:
            # Small-scale Rayleigh pour ce RB
            H_ss = draw_small_scale(K=K, M=M, rng=rng)   # [K, M]
            # Canal effectif : sqrt(beta_u) * h_{u}
            H_eff_all = (np.sqrt(beta)[:, None].astype(np.complex64)) * H_ss  # [K, M]

        if mode == "mrt_single":
            # On sert un seul utilisateur : celui avec le plus grand alpha*beta
            scores = alpha * beta
            u_star = int(np.argmax(scores))
            idx = np.array([u_star], dtype=int)
        elif mode == "zf_random":
            idx = rng.choice(K, size=min(U_max, K), replace=False)
        elif mode in ("zf_beta", "wmmse_alpha_beta"):
            if mode == "zf_beta":
                scores = beta
            else:
                scores = alpha * beta
            idx = select_top_k(scores, U_max)
        else:
            raise ValueError(f"Mode P3 inconnu : {mode}")

        H_sel = H_eff_all[idx, :]     # [U_n, M]
        alpha_sel = alpha[idx]        # [U_n]
        U_n = H_sel.shape[0]
        P_rb = P_RB_max

        if mode.startswith("zf"):
            # ZF + puissance égale
            sinr_rb = sinr_zf_equal_power(H_sel, P_tot=P_rb, sigma2=sigma2)  # [U_n]
        elif mode == "wmmse_alpha_beta":
            # WMMSE pondéré
            V, p = wmmse_precoder(H_sel, alpha_sel, P_tot=P_rb, sigma2=sigma2)
            sinr_rb = compute_sinr_given_beams(H_sel, V, p, sigma2)
        elif mode == "mrt_single":
            # MRT single-user (upper bound)
            h = H_sel[0, :]                    # [M]
            v = h.conj()
            v = v / (np.linalg.norm(v) + 1e-12)
            V = v.reshape(1, -1)               # [1, M]
            p = np.array([P_rb], dtype=np.float64)
            sinr_rb = compute_sinr_given_beams(H_sel, V, p, sigma2)
        else:
            raise ValueError(f"Mode P3 inconnu : {mode}")

        # Weighted sum-rate sur ce RB
        WSR_slot += float(np.sum(alpha_sel * np.log2(1.0 + sinr_rb)))

    return WSR_slot


# =====================================================================
# Boucle SNR -> moyenne WSR
# =====================================================================

def evaluate_p3_wsr_at_snr(cfg: SystemConfig,
                           snr_db: float,
                           n_slots: int = 1000,
                           seed: int = 1234,
                           mode: str = "zf_beta") -> tuple[float, float]:
    """
    Évalue la WSR moyenne de P3 pour un SNR donné.

    On impose :   P_RB_max / sigma2 = SNR_lin
      => sigma2 = P_RB_max / 10^(SNR_dB/10)
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma2 = cfg.P_RB_max / snr_lin

    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)

    # Pre-generate Sionna UMa channels at RB level: [B,K,N_RB,M]
    h_rb_true = generate_h_rb_true(cfg, batch_size=n_slots)
    h_rb_true_np = h_rb_true.numpy()

    vals = np.zeros(n_slots, dtype=np.float64)
    for t in range(n_slots):
        vals[t] = simulate_one_slot_p3(
            cfg, rng,
            mode=mode,
            sigma2_override=sigma2,
            h_rb_override=h_rb_true_np[t]
        )
    return float(vals.mean()), float(vals.std())


# =====================================================================
# Exécution directe : test rapide
# =====================================================================

if __name__ == "__main__":
    cfg = SystemConfig()
    rng = np.random.default_rng(0)

    print("=== P3 WSR evaluation (ZF, equal power) ===")
    print(f"M={cfg.M}, K={cfg.K}, N_RB={cfg.N_RB}, U_max={cfg.U_max}")
    print(f"Noise power sigma^2 = {cfg.sigma2:.3e} W")
    print(f"Per-RB power cap P_RB_max = {cfg.P_RB_max:.2f} W\n")

    wsr_vals = []
    for _ in range(1000):
        wsr_vals.append(simulate_one_slot_p3(cfg, rng, mode="zf_beta"))
    wsr_vals = np.array(wsr_vals)
    print(f"Average WSR over 1000 slots : {wsr_vals.mean():.3f} bit/s/Hz")
    print(f"Std of WSR : {wsr_vals.std():.3f} bit/s/Hz")
