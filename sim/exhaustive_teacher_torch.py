# sim/exhaustive_teacher_torch.py
import itertools
import torch
from typing import Optional, Tuple

from sim.sinr_balance_torch import sinr_balancing_power_constraint_torch

def best_group_by_balanced_sinr_torch(
    Hn: torch.Tensor,          # [M,K] complex64 on GPU
    sigma2: float,
    P_rb: float,
    U_max: int,
    rho: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      best_sel:  (U,) long
      beams:     (M,K) complex64  (unit-norm directions)
      powers:    (K,) float32
      sinr_u:    (K,) float32
      best_score:(,) float32 (linear)
    """
    device = Hn.device
    M, K = Hn.shape
    U_cap = min(U_max, M, K)

    best_score = torch.tensor(-1e30, device=device, dtype=torch.float32)
    best_sel = None
    best_Wdl = None
    best_sinr = None

    users = list(range(K))

    for U in range(1, U_cap + 1):
        for comb in itertools.combinations(users, U):
            sel = torch.tensor(comb, device=device, dtype=torch.long)
            H_sel = Hn[:, sel]  # [M,U]

            rho_u = torch.ones(U, device=device, dtype=torch.float32) if rho is None else rho[:U].to(device)

            Wdl, _, sinr_dl, _, _ = sinr_balancing_power_constraint_torch(
                H=H_sel, Pmax=float(P_rb), rho=rho_u, sigma2=float(sigma2)
            )
            score = torch.min(sinr_dl)
            if score > best_score:
                best_score = score
                best_sel = sel
                best_Wdl = Wdl
                best_sinr = sinr_dl

    beams = torch.zeros((M, K), device=device, dtype=torch.complex64)
    powers = torch.zeros((K,), device=device, dtype=torch.float32)
    sinr_u = torch.zeros((K,), device=device, dtype=torch.float32)

    if best_sel is not None:
        for j in range(best_sel.numel()):
            u = int(best_sel[j].item())
            v = best_Wdl[:, j]
            norm = torch.linalg.norm(v) + 1e-12
            beams[:, u] = v / norm
            powers[u] = (norm**2).real.to(torch.float32)
            sinr_u[u] = best_sinr[j]

    return best_sel, beams, powers, sinr_u, best_score
