# scripts/debug_p3_alpha.py
import numpy as np
from config import SystemConfig
from sim.channels_3gpp import gen_large_scale_3gpp_uma
from sim.weights import compute_alpha_from_beta

if __name__ == "__main__":
    cfg = SystemConfig()
    rng = np.random.default_rng(cfg.seed)

    beta, is_los, d2d = gen_large_scale_3gpp_uma(
        K=cfg.K,
        fc_GHz=cfg.fc_GHz,
        h_bs=cfg.h_bs,
        h_ut=cfg.h_ut,
        los_mode=cfg.los_mode,
        cell_radius_m=cfg.cell_radius_m,
        rng=rng,
    )

    alpha = compute_alpha_from_beta(beta)

    print("u | distance(m) | beta_u (lin)   | alpha_u")
    print("--------------------------------------------")
    for u in range(cfg.K):
        print(f"{u:2d} | {d2d[u]:9.2f} | {beta[u]:.3e} | {alpha[u]:.3f}")

    print("\nSum alpha_u =", alpha.sum())
