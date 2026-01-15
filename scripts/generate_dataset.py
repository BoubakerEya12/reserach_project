# scripts/generate_dataset.py
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from typing import Tuple
import numpy as np

from config import SystemConfig
from sim.scheduler import schedule_and_power
from sim.sinr_balance import sinr_balancing_power_constraint

# Optional teachers/solvers
try:
    from sim.exhaustive_teacher import best_group_by_balanced_sinr
    HAS_EXH = True
except Exception:
    HAS_EXH = False

try:
    from sim.wmmse import wmmse_sumrate
    HAS_WMMSE = True
except Exception:
    HAS_WMMSE = False

# Optional channel + pathloss modules
try:
    from sim.channels import make_channels
    HAS_CHANNELS_MODULE = True
except Exception:
    HAS_CHANNELS_MODULE = False

try:
    from sim.pathloss import log_distance_shadowing
    HAS_PATHLOSS_MODULE = True
except Exception:
    HAS_PATHLOSS_MODULE = False

# Optional 3GPP channel helper
_HAS_3GPP = False
try:
    # expected to provide: make_3gpp_channels(M,K,N_RB,scenario,fc_Hz,scs_Hz,n_sym_slot,eta,rng)
    from sim.channels_3gpp import make_3gpp_channels
    _HAS_3GPP = True
except Exception:
    _HAS_3GPP = False

# ---------- fallbacks ----------
def _fallback_log_distance_shadowing(
    rng: np.random.Generator,
    K: int,
    fc_GHz: float,
    d_min_m: float = 30.0,
    d_max_m: float = 500.0,
    alpha: float = 3.5,
    shadowing_sigma_dB: float = 6.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
    r2 = rng.uniform(d_min_m**2, d_max_m**2, size=K)
    d_m = np.sqrt(r2)
    fspl_1m_dB = 32.4 + 20.0 * np.log10(fc_GHz)
    X_sigma = rng.normal(loc=0.0, scale=shadowing_sigma_dB, size=K)
    PL_dB = fspl_1m_dB + 10.0 * alpha * np.log10(d_m / 1.0) + X_sigma
    beta = 10.0 ** (-PL_dB / 10.0)
    return beta.astype(np.float32), d_m.astype(np.float32)

def _fallback_make_channels(M: int, K: int, N_RB: int,
                            beta: np.ndarray, eta: float,
                            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    re = rng.normal(size=(K, N_RB, M))
    im = rng.normal(size=(K, N_RB, M))
    h = (re + 1j*im) / np.sqrt(2.0)
    h = h.astype(np.complex64)
    h_eff = np.sqrt(beta.reshape(K,1,1)).astype(np.float32) * h
    h_eff = h_eff.astype(np.complex64)
    re2 = rng.normal(size=(K, N_RB, M))
    im2 = rng.normal(size=(K, N_RB, M))
    e = (re2 + 1j*im2) / np.sqrt(2.0)
    scale = np.sqrt(max(0.0, 1.0 - eta**2))
    h_hat = (eta * h_eff) + (scale * e.astype(np.complex64))
    return h_eff.astype(np.complex64), h_hat.astype(np.complex64)

# ---------- helpers ----------
def build_labels_from_powers(powers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N_RB, K = powers.shape
    Z = (powers > 0).astype(np.float32)
    phi = np.zeros_like(powers, dtype=np.float32)
    Pn = np.sum(powers, axis=1, keepdims=True)
    active = Pn[:, 0] > 0
    if np.any(active):
        phi[active] = (powers[active] / Pn[active]).astype(np.float32)
    return Z, phi

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Generate MU-MIMO-OFDM dataset + labels.")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--out", type=str, default="dataset.npz")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--labeler", type=str,
                        choices=["zf", "sinr_balance", "exhaustive_p1", "wmmse_wsr"],
                        default="zf",
                        help="Label generator")
    parser.add_argument("--beamformer", type=str, choices=["zf","rzf"], default="rzf",
                        help="Only used with labeler=zf.")

    # >>> NEW: channel selector & CSIT control <<<
    parser.add_argument("--channel", choices=["rayleigh","umi","uma"], default="rayleigh",
                        help="Channel model used to draw (h_true, h_hat).")
    parser.add_argument("--fc-GHz", type=float, default=3.5,
                        help="Carrier frequency for path loss / 3GPP (GHz).")
    parser.add_argument("--eta", type=float, default=1.0,
                        help="CSIT quality in [0,1] (also used by 3GPP helper if present).")

    args = parser.parse_args()

    cfg = SystemConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    rng = np.random.default_rng(cfg.seed)

    # Let CLI override config (light-touch)
    if args.fc_GHz is not None:
        cfg.fc_GHz = float(args.fc_GHz)
    if args.eta is not None:
        cfg.eta = float(args.eta)

    M = cfg.M
    K = cfg.K
    N_RB = cfg.N_RB
    U_max = cfg.U_max
    sigma2 = cfg.noise_power()

    H_true_list, H_hat_list = [], []
    Z_list, alpha_list, phi_list = [], [], []
    beams_list, powers_list, sinr_list = [], [], []
    meta_list = []

    for s in range(args.num_samples):
        # Large + small scale + CSIT
        if args.channel == "rayleigh":
            # Rayleigh with log-distance shadowing for beta (if available)
            if HAS_PATHLOSS_MODULE:
                beta, d_m = log_distance_shadowing(rng=rng, K=K, fc_GHz=cfg.fc_GHz)
            else:
                beta, d_m = _fallback_log_distance_shadowing(rng=rng, K=K, fc_GHz=cfg.fc_GHz)

            if HAS_CHANNELS_MODULE:
                h_eff_true, h_eff_hat = make_channels(M=M, K=K, N_RB=N_RB, beta=beta, eta=cfg.eta, rng=rng)
            else:
                h_eff_true, h_eff_hat = _fallback_make_channels(M=M, K=K, N_RB=N_RB, beta=beta, eta=cfg.eta, rng=rng)

        else:
            # 3GPP UMi/UMa if helper exists; otherwise fallback to Rayleigh + pathloss
            if _HAS_3GPP:
                h_eff_true, h_eff_hat = make_3gpp_channels(
                    M=M, K=K, N_RB=N_RB,
                    scenario=args.channel,
                    fc_Hz=cfg.fc_GHz*1e9,
                    scs_Hz=15e3,
                    n_sym_slot=1,
                    eta=cfg.eta,
                    rng=rng,
                )
                # create synthetic distances for metadata (optional)
                d_m = np.full((K,), 100.0, dtype=np.float32)
            else:
                if HAS_PATHLOSS_MODULE:
                    beta, d_m = log_distance_shadowing(rng=rng, K=K, fc_GHz=cfg.fc_GHz)
                else:
                    beta, d_m = _fallback_log_distance_shadowing(rng=rng, K=K, fc_GHz=cfg.fc_GHz)
                if HAS_CHANNELS_MODULE:
                    h_eff_true, h_eff_hat = make_channels(M=M, K=K, N_RB=N_RB, beta=beta, eta=cfg.eta, rng=rng)
                else:
                    h_eff_true, h_eff_hat = _fallback_make_channels(M=M, K=K, N_RB=N_RB, beta=beta, eta=cfg.eta, rng=rng)

        beams = np.zeros((N_RB, M, K), dtype=np.complex64)
        powers = np.zeros((N_RB, K), dtype=np.float32)
        sinr = np.zeros((N_RB, K), dtype=np.float32)
        Z = np.zeros((N_RB, K), dtype=np.float32)
        alpha = np.zeros((N_RB,), dtype=np.float32)
        phi = np.zeros((N_RB, K), dtype=np.float32)

        remaining_P = cfg.P_tot

        for n in range(N_RB):
            if remaining_P <= 0:
                continue
            P_rb = float(min(cfg.P_RB_max, remaining_P))
            if P_rb <= 0:
                continue

            # Scores pour ordre glouton (si besoin)
            hhat_rb = h_eff_hat[:, n, :]   # [K,M]
            scores = np.sum(np.abs(hhat_rb)**2, axis=1)
            order = np.argsort(-scores)

            if args.labeler == "zf":
                out = schedule_and_power(
                    h_eff=h_eff_true[:, n:n+1, :],
                    h_eff_hat=h_eff_hat[:, n:n+1, :],
                    sigma2=sigma2,
                    P_RB_max=P_rb,
                    P_tot=P_rb,
                    U_max=U_max,
                    beamformer=args.beamformer,
                )
                beams[n] = out["beams"][0]
                powers[n] = out["powers"][0]
                sinr[n] = out["sinr"][0]
                alpha[n] = out["alpha_used"][0]

            elif args.labeler == "sinr_balance":
                sel = order[:U_max]
                sel = sel[scores[sel] > 0]
                U = sel.size
                if U > 0:
                    Hn = h_eff_true[sel, n, :].T  # [M,U]
                    Wdl, Wul, sinr_u, q_ul, p_u = sinr_balancing_power_constraint(
                        H=Hn, Pmax=P_rb, rho=np.ones(U, dtype=float), sigma2=sigma2
                    )
                    for j, u in enumerate(sel):
                        dir_u = (Wdl[:, j] / (np.linalg.norm(Wdl[:, j]) + 1e-12)).astype(np.complex64)
                        beams[n, :, u] = dir_u
                        powers[n, u]   = np.float32(np.linalg.norm(Wdl[:, j])**2)
                        sinr[n, u]     = np.float32(sinr_u[j])

            elif args.labeler == "exhaustive_p1":
                if not HAS_EXH:
                    raise RuntimeError("sim.exhaustive_teacher introuvable.")
                Hn_full = h_eff_true[:, n, :].T  # [M,K]
                sel, b_n, p_n, sinr_n, score = best_group_by_balanced_sinr(
                    Hn=Hn_full, sigma2=sigma2, P_rb=P_rb, U_max=U_max
                )
                beams[n] = b_n
                powers[n] = p_n
                sinr[n] = sinr_n

            else:  # wmmse_wsr
                if not HAS_WMMSE:
                    raise RuntimeError("sim.wmmse introuvable.")
                sel = order[:U_max]
                sel = sel[scores[sel] > 0]
                U = sel.size
                if U > 0:
                    Hn = h_eff_true[sel, n, :].T  # [M,U]
                    W, p_u, sinr_u = wmmse_sumrate(H=Hn, sigma2=sigma2, Pmax=P_rb, iters=50)
                    for j, u in enumerate(sel):
                        dir_u = (W[:, j] / (np.linalg.norm(W[:, j]) + 1e-12)).astype(np.complex64)
                        beams[n, :, u] = dir_u
                        powers[n, u]   = np.float32(np.linalg.norm(W[:, j])**2)
                        sinr[n, u]     = np.float32(sinr_u[j])

            remaining_P -= P_rb
            if remaining_P < 0:
                remaining_P = 0.0

        # Z/phi
        Z, phi = build_labels_from_powers(powers)

        # Store
        H_true_list.append(h_eff_true)
        H_hat_list.append(h_eff_hat)
        Z_list.append(Z)
        alpha_list.append(alpha)
        phi_list.append(phi)
        beams_list.append(beams)
        powers_list.append(powers)
        sinr_list.append(sinr)
        meta_list.append(dict(sample_index=int(s), rng_seed=int(cfg.seed),
                              sigma2=float(sigma2),
                              distances_m=d_m.tolist() if 'd_m' in locals() else [],
                              labeler=args.labeler,
                              channel=args.channel))

    # Stack + save
    H_true = np.stack(H_true_list, axis=0)
    H_hat  = np.stack(H_hat_list,  axis=0)
    Z      = np.stack(Z_list,      axis=0)
    alpha  = np.stack(alpha_list,  axis=0)
    phi    = np.stack(phi_list,    axis=0)
    beams  = np.stack(beams_list,  axis=0)
    powers = np.stack(powers_list, axis=0)
    sinr   = np.stack(sinr_list,   axis=0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cfg_json = json.dumps(asdict(cfg), indent=2)

    np.savez_compressed(
        args.out,
        H_true=H_true, H_hat=H_hat,
        Z=Z, alpha=alpha, phi=phi,
        beams=beams, powers=powers, sinr=sinr,
        sigma2=np.float32(sigma2),
        config_json=np.array(cfg_json),
        meta=np.array([json.dumps(m) for m in meta_list], dtype=object),
        labeler=np.array(args.labeler),
        beamformer=np.array(args.beamformer),
    )

    print("Saved:", args.out)
    print("Shapes:")
    print(f"  H_true : {H_true.shape}  (complex64)  -> [S, K, N_RB, M]")
    print(f"  H_hat  : {H_hat.shape}   (complex64)")
    print(f"  Z      : {Z.shape}       (float32)    -> [S, N_RB, K]")
    print(f"  alpha  : {alpha.shape}   (float32)    -> [S, N_RB]")
    print(f"  phi    : {phi.shape}     (float32)    -> [S, N_RB, K]")
    print(f"  beams  : {beams.shape}   (complex64)  -> [S, N_RB, M, K]")
    print(f"  powers : {powers.shape}  (float32)    -> [S, N_RB, K]")
    print(f"  sinr   : {sinr.shape}    (float32)    -> [S, N_RB, K]")
    print("Config snapshot:", cfg_json)
    print("Labeler:", args.labeler, "Beamformer:", args.beamformer, "Channel:", args.channel)

if __name__ == "__main__":
    main()
