
"""
Common simulation utilities for MU-MISO experiments (P1/P2/P3).

Numerical robustness:
- All Hermitian solves use complex128
- Diagonal jitter on Gram/T matrices when needed
- Robust solver with lstsq fallback
"""
import numpy as np
import itertools

# =========================
# RNG
# =========================
def make_rng(seed=1234):
    return np.random.default_rng(seed)

# =========================
# Channels
# =========================
def draw_iid_small_scale(rng, K, M):
    re = rng.normal(size=(K, M))
    im = rng.normal(size=(K, M))
    return ((re + 1j * im) / np.sqrt(2.0)).astype(np.complex64)

def draw_large_scale_beta(rng, K, fc_GHz=3.5,
                          d_min_m=30.0, d_max_m=500.0,
                          alpha=3.5, shadowing_sigma_dB=6.0):
    r2 = rng.uniform(d_min_m**2, d_max_m**2, size=K)
    d_m = np.sqrt(r2)
    fspl_1m_dB = 32.4 + 20.0 * np.log10(fc_GHz)
    Xsigma = rng.normal(0.0, shadowing_sigma_dB, size=K)
    PL_dB = fspl_1m_dB + 10.0 * alpha * np.log10(d_m / 1.0) + Xsigma
    beta = 10.0 ** (-PL_dB / 10.0)
    return beta.astype(np.float32)

def effective_channel(Hs, beta=None):
    # Hs: [K,M], beta: [K]
    if beta is None:
        return Hs.astype(np.complex64)
    K = Hs.shape[0]
    return (np.sqrt(beta).reshape(K, 1) * Hs).astype(np.complex64)

# =========================
# Robust Hermitian solver
# =========================
def _hermitian_solve(T, b, eps=None):
    """
    Solve T x = b with Hermitian T (complex).
    Adds small diagonal regularization if needed and falls back to lstsq.
    """
    M = T.shape[0]
    T = T.astype(np.complex128, copy=False)
    b = b.astype(np.complex128, copy=False)
    if eps is None:
        tr = np.real(np.trace(T))
        scale = tr / M if np.isfinite(tr) and tr > 0 else 1.0
        eps = 1e-12 * scale
    Tj = T + eps * np.eye(M, dtype=np.complex128)
    try:
        return np.linalg.solve(Tj, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(Tj, b, rcond=None)
        return x

# =========================
# Precoding baselines
# =========================
def zf_equal_power(H, Pmax):
    """
    ZF with equal power across active users.
    H: [U,M] user-rows; returns Wtx [M,U]
    """
    U, M = H.shape
    HH = H.astype(np.complex128, copy=False)
    Gram = HH @ HH.conj().T + 1e-12 * np.eye(U)
    W = HH.conj().T @ np.linalg.pinv(Gram)
    V = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    p_each = Pmax / max(U, 1)
    return V * np.sqrt(p_each)

def rzf_equal_power(H, sigma2, Pmax):
    """
    RZF with equal power across active users.
    """
    U, M = H.shape
    HH = H.astype(np.complex128, copy=False)
    alpha = U * float(sigma2) / max(float(Pmax), 1e-12)
    Gram = HH @ HH.conj().T + alpha * np.eye(U)
    W = HH.conj().T @ np.linalg.pinv(Gram)
    V = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    p_each = Pmax / max(U, 1)
    return V * np.sqrt(p_each)

# =========================
# Metrics
# =========================
def sinr_vector(H, Wtx, sigma2):                                 
    """
    H: [U,M], Wtx: [M,U]  -> Returns per-user SINR (linear)
    """
    U = H.shape[0]
    HW = (H.astype(np.complex128, copy=False)) @ (Wtx.astype(np.complex128, copy=False))
    sig = np.abs(np.diag(HW)) ** 2
    interf = np.sum(np.abs(HW) ** 2, axis=1) - sig
    return sig / (interf + float(sigma2))

def min_sinr_db(H, Wtx, sigma2):
    return 10.0 * np.log10(max(1e-30, np.min(sinr_vector(H, Wtx, sigma2))))

# =========================
# P1 Optimal (uplink–downlink duality with bisection on γ)
# =========================
def p1_maxmin_sinr_db(H, sigma2, Pmax, tol_db=1e-3, max_iter=60, q0=None):
    """
    Max–min balanced SINR (in dB) for a fixed user set (rows of H).
    Robust fixed-point feasibility + bisection on γ (no channel scaling).
    """
    Hc = H.astype(np.complex128, copy=False)   # [K,M]
    K, M = Hc.shape
    sigma2 = float(sigma2)
    Pmax   = float(Pmax)
    I = np.eye(M, dtype=np.complex128)

    def total_uplink_power_for_gamma(gamma_lin, q_init=None):
        """
        Given target γ (linear), run the standard fixed-point for the virtual
        uplink powers q to test feasibility. Returns (sum(q), q).
        """
        if (q_init is None) or (np.size(q_init) != K):
            q = np.full(K, Pmax / max(K, 1), dtype=np.float64)
        else:
            q = np.array(q_init, dtype=np.float64, copy=True)
        q[q < 1e-15] = 1e-15

        for _ in range(max_iter):
            # T = σ² I + Σ_i q_i h_i h_i^H
            T = sigma2 * I.copy()
            for i in range(K):
                hi = Hc[i, :].reshape(M, 1)
                T += q[i] * (hi @ hi.conj().T)

            # Try Cholesky; fallback to robust Hermitian solve
            try:
                L = np.linalg.cholesky(T + 1e-12 * I)
                def solve_T(b):
                    y = np.linalg.solve(L, b)
                    return np.linalg.solve(L.conj().T, y)
            except np.linalg.LinAlgError:
                def solve_T(b):
                    return _hermitian_solve(T + 1e-12 * I, b)

            # g_k = h_k^H T^{-1} h_k
            g = np.empty(K, dtype=np.float64)
            for k in range(K):
                hk = Hc[k, :].reshape(M, 1)
                x  = solve_T(hk)
                gk = float(np.real((hk.conj().T @ x)[0, 0]))
                g[k] = max(gk, 1e-20)

            q_new = gamma_lin / g
            # **Slightly stronger relaxation** for near-symmetric channels
            q = 0.7 * q + 0.3 * q_new

            # **Looser relative tolerance** improves convergence stability
            if np.max(np.abs(q_new - q) / np.maximum(q, 1e-12)) < 5e-4:
                break

        return float(np.sum(q)), q

    # ---- bisection on γ (linear scale)
    # **Wider initial bracket** works better for small-scale only cases
    lo, hi = 1e-10, 1e-1
    q_warm = q0

    # Expand hi until infeasible (or cap)
    for _ in range(30):
        tot, q_warm = total_uplink_power_for_gamma(hi, q_warm)
        if tot <= Pmax:
            hi *= 2.0
            if hi > 1e3:   # safety stop
                break
        else:
            break

    # Geometric bisection
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        tot, q_warm = total_uplink_power_for_gamma(mid, q_warm)
        if tot <= Pmax:
            lo = mid
        else:
            hi = mid
        if 10.0 * np.log10(hi / lo) < tol_db:
            break

    gamma_star = lo
    return 10.0 * np.log10(max(gamma_star, 1e-30))



# ---------- pruning upper bound for exhaustive ----------
def _ub_gamma_subset(H_sub, sigma2, Pmax):
    """
    Valid upper bound on balanced SINR for a subset S:
        Σ p_k >= γ σ² Σ (1/||h_k||²)  and  Σ p_k <= Pmax
      ⇒ γ* <= Pmax / (σ² * Σ_k 1/||h_k||²).
    Works even with interference (necessary condition).
    """
    Hc = H_sub.astype(np.complex128, copy=False)
    norms2 = np.sum(np.abs(Hc)**2, axis=1)
    inv_sum = np.sum(1.0 / np.maximum(norms2, 1e-18))
    return float(Pmax / (float(sigma2) * inv_sum + 1e-18))

def exhaustive_p1_over_subsets(H, sigma2, Pmax, Umax=2, tol_db=1e-3):
    """
    Exhaustive over user subsets of size <= Umax (for P1) with pruning.
    Returns (best_minSINR_dB, best_subset_indices)
    - Visits users in strength order to obtain good early bounds.
    - Prunes subset S if UB(S) <= best_gamma_found.
    """
    K, _ = H.shape
    users = np.arange(K, dtype=int)
    strengths = np.sum(np.abs(H)**2, axis=1)
    order = users[np.argsort(-strengths)]

    best_gamma = 0.0  # linear
    best_sel = None

    for U in range(1, min(int(Umax), K) + 1):
        for sel in itertools.combinations(order, U):
            idx = np.array(sel, dtype=int)
            HH = H[idx, :]

            ub = _ub_gamma_subset(HH, sigma2, Pmax)
            if ub <= best_gamma * (1.0 + 1e-6):
                continue

            try:
                val_db = p1_maxmin_sinr_db(HH, sigma2, Pmax, tol_db=tol_db)
            except np.linalg.LinAlgError:
                continue

            val_lin = 10.0**(val_db / 10.0)
            if val_lin > best_gamma:
                best_gamma = val_lin
                best_sel = tuple(idx)

    best_db = 10.0 * np.log10(max(best_gamma, 1e-30))
    return best_db, best_sel

# =========================
# P2 (power minimization under SINR targets Γ)
# =========================
def p2_min_power(H, gamma_vec, sigma2):
    """
    Returns (total_power, p>=0, Wtx) or (np.inf, None, None) if infeasible.
    Robust implementation using Hermitian solves; avoids explicit inverses.
    """
    Hc = H.astype(np.complex128, copy=False)
    U, M = Hc.shape
    sigma2 = float(sigma2)
    gamma = np.array(gamma_vec, dtype=np.float64, copy=False)

    # 1) virtual uplink fixed-point to get normalized downlink directions
    q = np.ones(U, dtype=np.float64)
    for _ in range(300):
        T = sigma2 * np.eye(M, dtype=np.complex128)
        for i in range(U):
            hi = Hc[i, :].reshape(M, 1)
            T += q[i] * (hi @ hi.conj().T)

        g = np.empty(U, dtype=np.float64)
        for k in range(U):
            hk = Hc[k, :].reshape(M, 1)
            x = _hermitian_solve(T, hk)
            gk = float(np.real((hk.conj().T @ x)[0, 0]))
            g[k] = max(gk, 1e-18)

        q_new = gamma / g
        if np.max(np.abs(q_new - q) / np.maximum(q, 1e-12)) < 1e-4:
            q = q_new
            break
        q = 0.5 * q + 0.5 * q_new

    # normalized beam directions V (downlink)
    T = sigma2 * np.eye(M, dtype=np.complex128)
    for i in range(U):
        hi = Hc[i, :].reshape(M, 1)
        T += q[i] * (hi @ hi.conj().T)

    V = np.zeros((M, U), dtype=np.complex128)
    for k in range(U):
        hk = Hc[k, :].conj()  # RHS is conj(h_k) for transmit direction
        vk = _hermitian_solve(T, hk)
        nrm = np.linalg.norm(vk)
        if nrm < 1e-15:
            return np.inf, None, None
        V[:, k] = vk / nrm

    # 2) Solve for downlink powers p from Ψ p = σ² 1
    Psi = np.zeros((U, U), dtype=np.float64)
    for k in range(U):
        hk = Hc[k, :]
        for j in range(U):
            hV = np.abs(hk @ V[:, j]) ** 2
            if j == k:
                Psi[k, k] = hV / max(gamma[k], 1e-18)
            else:
                Psi[k, j] = -hV
    Psi = Psi + 1e-12 * np.eye(U)
    try:
        p = sigma2 * np.linalg.solve(Psi, np.ones(U))
    except np.linalg.LinAlgError:
        return np.inf, None, None

    if np.any(p < -1e-9):
        return np.inf, None, None

    p = np.maximum(p, 0.0)
    Wtx = V * np.sqrt(p.reshape(1, -1))
    return float(np.sum(p)), p, Wtx

# =========================
# P3 (sum-rate) via WMMSE
# =========================
def wmmse_sumrate(H, sigma2, Pmax, iters=50):
    """
    Standard WMMSE loop with robust steps and total-power normalization.
    Returns (sum_rate_bits, Wtx)
    """
    Hc = H.astype(np.complex128, copy=False)
    U, M = Hc.shape
    sigma2 = float(sigma2)
    Pmax = float(Pmax)

    # init: RZF equal power (robust)
    W = rzf_equal_power(Hc, sigma2, Pmax)  # [M,U]

    for _ in range(iters):
        # Receivers and MSE weights
        HW = Hc @ W  # [U,U]
        diag_HW = np.diag(HW)
        power_tot = np.sum(np.abs(HW) ** 2, axis=1) + sigma2
        interf = power_tot - np.abs(diag_HW) ** 2
        g = diag_HW / np.maximum(interf, 1e-18)         # MMSE receivers
        e = 1 - 2 * np.real(g * diag_HW) + (np.abs(g) ** 2) * power_tot
        w = 1.0 / np.maximum(e, 1e-12)

        # Update transmit W (quadratic form); solve (A + mu I)W = B
        A = np.zeros((M, M), dtype=np.complex128)
        B = np.zeros((M, U), dtype=np.complex128)
        for k in range(U):
            hk = Hc[k, :].reshape(M, 1)
            A += w[k] * (np.abs(g[k]) ** 2) * (hk @ hk.conj().T)
            B[:, k] = w[k] * g[k].conj() * Hc[k, :].conj()

        Aj = A + 1e-12 * np.eye(M)
        W_tmp = np.linalg.pinv(Aj) @ B

        # Enforce total power
        curP = float(np.real(np.sum(np.linalg.norm(W_tmp, axis=0) ** 2)))
        if curP > Pmax:
            W = W_tmp * np.sqrt(Pmax / (curP + 1e-12))
        else:
            W = W_tmp

    snr = sinr_vector(Hc, W, sigma2)
    Rsum = float(np.sum(np.log2(1.0 + snr)))
    return Rsum, W



# scripts/common.py  (add near the P1 section)
def p1_solve_fixed_users_tuple(H, sigma2, Pmax):
    """
    P1 (max–min SINR) with fixed users = rows(H).
    Returns (W, p, sinr, gamma_db), where W is [M,U].
    """
    gamma_db = p1_maxmin_sinr_db(H, sigma2, Pmax)
    gamma_lin = 10.0**(gamma_db/10.0)
    totP, p, W = p2_min_power(H, np.full(H.shape[0], gamma_lin), sigma2)
    if not np.isfinite(totP) or W is None:
        # infeasible (very low SNR): return dummies
        U = H.shape[0]
        return (np.zeros((H.shape[1], U), dtype=np.complex64),
                np.zeros((U,), dtype=np.float32),
                np.zeros((U,), dtype=np.float32),
                -300.0)
    sinr = sinr_vector(H, W, sigma2).astype(np.float32)
    p = (np.linalg.norm(W, axis=0)**2).astype(np.float32)
    return W.astype(np.complex64), p, sinr, float(gamma_db)
