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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--sigma2", type=float, default=1e-9)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--fc_GHz", type=float, default=3.5)
    ap.add_argument("--Gamma_dB", type=float, nargs="+", default=[-5, 0, 5, 10, 15])
    ap.add_argument("--include_large", action="store_true", help="large+small fading if set")
    ap.add_argument("--out", type=str, default="p2_power_vs_gamma.png")
    args = ap.parse_args()

    rng = make_rng(1234)
    gammas = np.array([10.0**(g/10.0) for g in args.Gamma_dB], dtype=np.float64)

    meanP = []
    feas_ratio = []
    for g_lin, g_db in zip(gammas, args.Gamma_dB):
        totP = 0.0
        feas = 0
        gamma_vec = np.full(args.K, g_lin, dtype=np.float64)
        for _ in range(args.samples):
            Hs = draw_iid_small_scale(rng, args.K, args.M)
            if args.include_large:
                beta = draw_large_scale_beta(rng, args.K, args.fc_GHz)
                H = effective_channel(Hs, beta)
            else:
                H = Hs
            Ptot, p, Wtx = p2_min_power(H[:min(args.K,args.M), :], gamma_vec[:min(args.K,args.M)], args.sigma2)
            if np.isfinite(Ptot):
                totP += Ptot; feas += 1
        meanP.append(totP/max(feas,1))
        feas_ratio.append(feas/args.samples)

    plt.figure(figsize=(7.2,5.0))
    plt.plot(args.Gamma_dB, 10.0*np.log10(np.maximum(np.array(meanP)/args.sigma2, 1e-30)),
             "o-", label="Avg required P / σ² (dB)")
    plt.xlabel("SINR target Γ (dB)")
    plt.ylabel(r"Required power $10\log_{10}(P_{\rm req}/\sigma^2)$ (dB)")
    title = "(with large+small)" if args.include_large else "(small-scale only)"
    plt.title(f"P2: power vs SINR target Γ  {title}  |  K={args.K}, M={args.M}, samples={args.samples}")
    plt.grid(True, alpha=0.3)
    plt.twinx()
    plt.plot(args.Gamma_dB, np.array(feas_ratio)*100.0, "s--", label="Feasibility (%)", color="tab:orange")
    plt.ylabel("Feasible cases (%)")
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(args.out, dpi=220)

if __name__ == "__main__":
    main()
