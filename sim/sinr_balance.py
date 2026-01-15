# sim/sinr_balance.py
import numpy as np

def sinr_balancing_power_constraint(H: np.ndarray, Pmax: float, rho: np.ndarray, sigma2: float,
                                    tol: float = 1e-5, max_iter: int = 200):
    """
    SINR balancing under total power, via uplink–downlink duality (Algorithm 1).
    Args:
        H:      (N, K) complex ndarray, columns are user channels h_k
        Pmax:   float, total downlink power constraint
        rho:    (K,) balancing weights (all ones => equal)
        sigma2: float, noise power
    Returns:
        w_dl:   (N, K) complex, downlink beamformers (already scaled by sqrt(p_k))
        w_ul:   (N, K) complex, unit-norm UL combiners (directions)
        sinr_dl:(K,) float, achieved DL SINRs (nearly equal when balanced)
        q_ul:   (K,) float, final UL powers
        p_dl:   (K,) float, final DL powers (sum <= Pmax)
    """
    N, K = H.shape
    rho = np.asarray(rho, dtype=float).reshape(-1)
    assert rho.shape[0] == K
    q = np.zeros(K, dtype=float)  # uplink powers
    w_ul = np.zeros((N, K), dtype=np.complex64)
    prev_ev = None

    for _ in range(max_iter):
        # T = sigma^2 I + sum_k q_k h_k h_k^H
        T = sigma2 * np.eye(N, dtype=np.complex64)
        for k in range(K):
            hk = H[:, k].reshape(-1, 1)
            T += q[k] * (hk @ hk.conj().T)

        # UL combiners w_ul(:,k) = T^{-1} h_k (unit-norm)
        T_inv = np.linalg.pinv(T)
        for k in range(K):
            wk = T_inv @ H[:, k]
            nk = np.linalg.norm(wk) + 1e-12
            w_ul[:, k] = (wk / nk).astype(np.complex64)

        # Build D and F
        D = np.zeros((K, K), dtype=float)
        F = np.zeros((K, K), dtype=float)
        for k in range(K):
            gkk = np.abs(w_ul[:, k].conj().T @ H[:, k])**2 / sigma2
            D[k, k] = rho[k] / max(gkk, 1e-20)
        for x in range(K):
            for k in range(K):
                if k == x:
                    continue
                gxk = np.abs(w_ul[:, x].conj().T @ H[:, k])**2 / sigma2
                F[k, x] = gxk

        one = np.ones((K, 1), dtype=float)

        # Eigen update for UL powers: X = [[D F^T, D*1],[1^T D F^T / Pmax, 1^T D 1 / Pmax]]
        blockA = D @ F.T
        blockB = D @ one
        blockC = (one.T @ D @ F.T) / max(Pmax, 1e-12)
        blockD = (one.T @ D @ one) / max(Pmax, 1e-12)
        X = np.block([[blockA, blockB],
                      [blockC, blockD]])

        evals, evecs = np.linalg.eig(X)
        idx = np.argmax(np.real(evals))
        ev = float(np.real(evals[idx]))
        v = np.real(evecs[:, idx])
        v = v / max(v[-1], 1e-20)
        q_new = np.maximum(v[:K], 0.0)

        if prev_ev is not None:
            rel = abs(ev - prev_ev) / (abs(prev_ev) + 1e-12)
            if rel < tol:
                q = q_new
                break
        q = q_new
        prev_ev = ev

    # DL powers from DF (not DF^T)
    blockA = D @ F
    blockB = D @ one
    blockC = (one.T @ D @ F) / max(Pmax, 1e-12)
    blockD = (one.T @ D @ one) / max(Pmax, 1e-12)
    X2 = np.block([[blockA, blockB],
                   [blockC, blockD]])
    evals, evecs = np.linalg.eig(X2)
    idx = np.argmax(np.real(evals))
    v = np.real(evecs[:, idx])
    v = v / max(v[-1], 1e-20)
    p = np.maximum(v[:K], 0.0)
    s = p.sum()
    if s > Pmax and s > 0:
        p *= (Pmax / s)

    # Downlink beams = sqrt(p_k) * UL directions
    w_dl = w_ul * np.sqrt(p.reshape(1, -1))

    # DL SINR
    sinr_dl = np.zeros(K, dtype=float)
    for k in range(K):
        sig = np.abs(H[:, k].conj().T @ w_dl[:, k])**2
        interf = 0.0
        for j in range(K):
            if j == k:
                continue
            interf += np.abs(H[:, k].conj().T @ w_dl[:, j])**2
        sinr_dl[k] = float(sig / (interf + sigma2))

    return w_dl.astype(np.complex64), w_ul.astype(np.complex64), \
           sinr_dl.astype(np.float32), q.astype(np.float32), p.astype(np.float32)
