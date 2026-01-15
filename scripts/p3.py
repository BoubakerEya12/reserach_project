# -*- coding: utf-8 -*-
import argparse, numpy as np
import matplotlib.pyplot as plt
# --- robust relative/absolute import (works with -m and direct run) ---
try:
    from .common import (make_rng, draw_iid_small_scale, draw_large_scale_beta,
                         effective_channel, zf_equal_power, rzf_equal_power,
                         min_sinr_db, p1_maxmin_sinr_db, exhaustive_p1_over_subsets)
except ImportError:
    # Fallback when the file is run directly (not as a module)
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from common import (make_rng, draw_iid_small_scale, draw_large_scale_beta,
                        effective_channel, zf_equal_power, rzf_equal_power,
                        min_sinr_db, p1_maxmin_sinr_db, exhaustive_p1_over_subsets)
# ----------------------------------------------------------------------

def run_once(H, sigma2, Pmax):
    U, M = H.shape
    # Baselines
    Usel = min(U,M)
    sel = np.argsort(-np.sum(np.abs(H)**2, axis=1))[:Usel]
    HH = H[sel,:]
    Wzf = zf_equal_power(HH, Pmax)
    Wr = rzf_equal_power(HH, sigma2, Pmax)
    Rzf = np.sum(np.log2(1.0 + sinr_vector(HH, Wzf, sigma2)))
    Rrzf = np.sum(np.log2(1.0 + sinr_vector(HH, Wr, sigma2)))
    # WMMSE
    Rwmmse, W = wmmse_sumrate(HH, sigma2, Pmax, iters=50)
    return Rzf, Rrzf, Rwmmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--sigma2", type=float, default=1e-9)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--fc_GHz", type=float, default=3.5)
    ap.add_argument("--include_large", action="store_true")
    ap.add_argument("--out", type=str, default="p3_sumrate.png")
    args = ap.parse_args()

    rng = make_rng(1234)
    snr_grid_db = np.arange(0, 31, 3)

    mean_zf, mean_rzf, mean_w = [], [], []
    for snr_db in snr_grid_db:
        Pmax = args.sigma2 * (10.0**(snr_db/10.0))
        acc = np.zeros(3, dtype=np.float64)
        for _ in range(args.samples):
            Hs = draw_iid_small_scale(rng, args.K, args.M)
            if args.include_large:
                beta = draw_large_scale_beta(rng, args.K, args.fc_GHz)
                H = effective_channel(Hs, beta)
            else:
                H = Hs
            r1,r2,r3 = run_once(H, args.sigma2, Pmax)
            acc += np.array([r1,r2,r3])
        mean_zf.append(acc[0]/args.samples)
        mean_rzf.append(acc[1]/args.samples)
        mean_w.append(acc[2]/args.samples)

    plt.figure(figsize=(7.6,5.0))
    plt.plot(snr_grid_db, mean_w, "o-", label="WMMSE (P3)")
    plt.plot(snr_grid_db, mean_zf, "^-", label="ZF (equal power)")
    plt.plot(snr_grid_db, mean_rzf, "s-", label="RZF (equal power)")
    title = "(with large+small)" if args.include_large else "(small-scale only)"
    plt.title(f"P3: average sum-rate vs SNR  {title}  |  K={args.K}, M={args.M}, samples={args.samples}")
    plt.xlabel(r"Normalized transmit power $10\log_{10}(P_{\max}/\sigma^2)$ (dB)")
    plt.ylabel("Average sum-rate (bit/s/Hz)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(args.out, dpi=220)

if __name__ == "__main__":
    main()
