# sim/sinr_balance_torch.py
import torch

def sinr_balancing_power_constraint_torch(
    H: torch.Tensor,          # (N,K) complex64 on GPU
    Pmax: float,
    rho: torch.Tensor,        # (K,) float on GPU
    sigma2: float,
    tol: float = 1e-5,
    max_iter: int = 200,
):
    """
    Torch/CUDA port of sinr_balancing_power_constraint (Algorithm 1).
    Returns:
      w_dl:   (N,K) complex64
      w_ul:   (N,K) complex64
      sinr_dl:(K,) float32
      q_ul:   (K,) float32
      p_dl:   (K,) float32
    """
    assert H.is_complex(), "H must be complex tensor"
    device = H.device
    N, K = H.shape

    rho = rho.reshape(-1).to(device=device, dtype=torch.float32)
    assert rho.numel() == K

    q = torch.zeros(K, device=device, dtype=torch.float32)
    w_ul = torch.zeros((N, K), device=device, dtype=torch.complex64)

    prev_ev = None
    one = torch.ones((K, 1), device=device, dtype=torch.float32)

    sigma2_c = torch.tensor(float(sigma2), device=device, dtype=torch.float32)
    Pmax_t = torch.tensor(float(Pmax), device=device, dtype=torch.float32)

    for _ in range(max_iter):
        # T = sigma^2 I + sum_k q_k h_k h_k^H
        T = (sigma2_c * torch.eye(N, device=device, dtype=torch.complex64))
        for k in range(K):
            hk = H[:, k].reshape(N, 1)
            T = T + q[k].to(torch.complex64) * (hk @ hk.conj().T)

        # UL combiners: w_ul(:,k) = T^{-1} h_k, then unit norm
        T_inv = torch.linalg.pinv(T)
        for k in range(K):
            wk = T_inv @ H[:, k]
            nk = torch.linalg.norm(wk) + 1e-12
            w_ul[:, k] = (wk / nk).to(torch.complex64)

        # Build D and F (float)
        D = torch.zeros((K, K), device=device, dtype=torch.float32)
        F = torch.zeros((K, K), device=device, dtype=torch.float32)

        for k in range(K):
            gkk = torch.abs(torch.vdot(w_ul[:, k], H[:, k]))**2 / sigma2_c
            D[k, k] = rho[k] / torch.clamp(gkk.real, min=1e-20)

        for x in range(K):
            wx = w_ul[:, x]
            for k in range(K):
                if k == x:
                    continue
                gxk = torch.abs(torch.vdot(wx, H[:, k]))**2 / sigma2_c
                F[k, x] = gxk.real

        # X = [[D F^T, D*1],[1^T D F^T / Pmax, 1^T D 1 / Pmax]]
        blockA = D @ F.T
        blockB = D @ one
        blockC = (one.T @ D @ F.T) / torch.clamp(Pmax_t, min=1e-12)
        blockD = (one.T @ D @ one) / torch.clamp(Pmax_t, min=1e-12)

        X = torch.cat([torch.cat([blockA, blockB], dim=1),
                       torch.cat([blockC, blockD], dim=1)], dim=0)

        evals, evecs = torch.linalg.eig(X.to(torch.complex64))
        evals_r = evals.real
        idx = torch.argmax(evals_r)
        ev = float(evals_r[idx].item())

        v = evecs[:, idx].real
        v = v / torch.clamp(v[-1], min=1e-20)
        q_new = torch.clamp(v[:K], min=0.0).to(torch.float32)

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
    blockC = (one.T @ D @ F) / torch.clamp(Pmax_t, min=1e-12)
    blockD = (one.T @ D @ one) / torch.clamp(Pmax_t, min=1e-12)

    X2 = torch.cat([torch.cat([blockA, blockB], dim=1),
                    torch.cat([blockC, blockD], dim=1)], dim=0)

    evals2, evecs2 = torch.linalg.eig(X2.to(torch.complex64))
    idx2 = torch.argmax(evals2.real)
    v2 = evecs2[:, idx2].real
    v2 = v2 / torch.clamp(v2[-1], min=1e-20)

    p = torch.clamp(v2[:K], min=0.0).to(torch.float32)
    s = torch.sum(p)
    if float(s.item()) > float(Pmax) and float(s.item()) > 0.0:
        p = p * (Pmax_t / s)

    # Downlink beams
    w_dl = w_ul * torch.sqrt(p.reshape(1, -1)).to(torch.complex64)

    # DL SINR
    sinr_dl = torch.zeros(K, device=device, dtype=torch.float32)
    for k in range(K):
        hk = H[:, k]
        sig = torch.abs(torch.vdot(hk, w_dl[:, k]))**2
        interf = 0.0
        for j in range(K):
            if j == k:
                continue
            interf = interf + torch.abs(torch.vdot(hk, w_dl[:, j]))**2
        sinr_dl[k] = (sig.real / (interf.real + sigma2_c)).to(torch.float32)

    return (w_dl.to(torch.complex64),
            w_ul.to(torch.complex64),
            sinr_dl.to(torch.float32),
            q.to(torch.float32),
            p.to(torch.float32))
