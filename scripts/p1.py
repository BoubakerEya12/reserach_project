# -*- coding: utf-8 -*-
import argparse, numpy as np
import matplotlib.pyplot as plt
import os, sys

# --- robust import (works both with `python -m` and direct run) ---
try:
    from .common import (
        make_rng, draw_iid_small_scale, draw_large_scale_beta,
        effective_channel, zf_equal_power, rzf_equal_power,
        min_sinr_db, p1_maxmin_sinr_db, exhaustive_p1_over_subsets,
        p2_min_power
    )
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from common import (
        make_rng, draw_iid_small_scale, draw_large_scale_beta,
        effective_channel, zf_equal_power, rzf_equal_power,
        min_sinr_db, p1_maxmin_sinr_db, exhaustive_p1_over_subsets,
        p2_min_power
    )

# ----------------------------------------------------------------------
def ensure_dir(path: str):
    """Create output directory if it doesn’t exist."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
def eval_case(K, M, sigma2, Pmax, include_large, rng, fc_GHz,
              exhaustive_Umax=None, check_p2=False):
    """
    One Monte Carlo realization for a given SNR and fading type.
    Returns (opt_db, zf_db, rzf_db, ex_db or None)
    """
    # small-scale
    Hs = draw_iid_small_scale(rng, K, M)
    # add large-scale if requested
    if include_large:
        beta = draw_large_scale_beta(rng, K, fc_GHz)
        H = effective_channel(Hs, beta)
    else:
        H = Hs

    # --- Optimal (teacher) over all K users ---
    opt_db = p1_maxmin_sinr_db(H, sigma2, Pmax)

    # Optional cross-check: verify P1’s gamma via P2 (diagnostic only)
    if check_p2:
        gamma_lin = 10.0 ** (opt_db / 10.0)
        totP, p_vec, Wtx = p2_min_power(H, np.full(H.shape[0], gamma_lin), sigma2)
        if np.isfinite(totP) and totP <= Pmax * 1.05 and Wtx is not None:
            opt_db_check = min_sinr_db(H, Wtx, sigma2)
            if abs(opt_db_check - opt_db) > 0.2:
                print(f"[WARN] P1/P2 mismatch: P1={opt_db:.2f} dB vs P2={opt_db_check:.2f} dB")

    # --- Equal-power ZF/RZF on best U = min(K,M) users by norm ---
    U = min(K, M)
    sel = np.argsort(-np.sum(np.abs(H) ** 2, axis=1))[:U]
    HH = H[sel, :]
    zf_db  = min_sinr_db(HH, zf_equal_power(HH, Pmax), sigma2)
    rzf_db = min_sinr_db(HH, rzf_equal_power(HH, sigma2, Pmax), sigma2)

    # --- Optional exhaustive search (over subsets up to Umax) ---
    ex_db = None
    if exhaustive_Umax is not None and exhaustive_Umax >= 1:
        ex_db, _ = exhaustive_p1_over_subsets(H, sigma2, Pmax, Umax=int(exhaustive_Umax))

    return opt_db, zf_db, rzf_db, ex_db

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--sigma2", type=float, default=1e-9)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--fc_GHz", type=float, default=3.5)

    ap.add_argument("--exhaustive_Umax", type=int, default=0,
                    help="if >0, run exhaustive over user subsets of size ≤ this")
    ap.add_argument("--check_p2", action="store_true",
                    help="diagnostic: cross-check P1 result with P2 at γ*")

    ap.add_argument("--out_a", type=str, default="p1_small_only.png")
    ap.add_argument("--out_b", type=str, default="p1_large+small.png")
    ap.add_argument("--verbose", action="store_true", help="print progress")
    ap.add_argument("--log_every", type=int, default=50,
                    help="when --verbose, print every N samples per SNR")
    args = ap.parse_args()

    rng = make_rng(1234)
    snr_grid_db = np.arange(0, 31, 3)

    # results dict: key 'a' for small-scale only, 'b' for large+small
    results = {}

    for include_large in [False, True]:
        key = "b" if include_large else "a"
        mean_opt, mean_zf, mean_rzf, mean_ex = [], [], [], []

        for snr_db in snr_grid_db:
            if args.verbose:
                tag = "LS+SS" if include_large else "SS-only"
                print(f"[{tag}] SNR={snr_db:.1f} dB → {args.samples} samples", flush=True)

            Pmax = args.sigma2 * (10.0 ** (snr_db / 10.0))
            acc_opt = 0.0
            acc_zf  = 0.0
            acc_rzf = 0.0
            acc_ex  = 0.0
            ex_count = 0  # because exhaustive may be None if disabled

            for s in range(args.samples):
                if args.verbose and (s % args.log_every == 0):
                    tag = "LS+SS" if include_large else "SS-only"
                    print(f"  [{tag}] sample {s}/{args.samples}", flush=True)

                opt_db, zf_db, rzf_db, ex_db = eval_case(
                    args.K, args.M, args.sigma2, Pmax,
                    include_large, rng, args.fc_GHz,
                    exhaustive_Umax=args.exhaustive_Umax,
                    check_p2=args.check_p2
                )
                acc_opt += opt_db
                acc_zf  += zf_db
                acc_rzf += rzf_db
                if ex_db is not None:
                    acc_ex  += ex_db
                    ex_count += 1

            mean_opt.append(acc_opt / args.samples)
            mean_zf.append(acc_zf / args.samples)
            mean_rzf.append(acc_rzf / args.samples)
            mean_ex.append((acc_ex / ex_count) if ex_count > 0 else None)

        results[key] = (
            np.array(mean_opt), np.array(mean_zf), np.array(mean_rzf), mean_ex
        )

    # -------------------- Plot (a): small-scale only --------------------
    t_a, z_a, r_a, ex_a = results["a"]
    plt.figure(figsize=(8.0, 5.2))
    plt.plot(snr_grid_db, t_a, "o-", label="Optimal P1 (max–min)")
    plt.plot(snr_grid_db, z_a, "^-", label="ZF (equal power, best U)")
    plt.plot(snr_grid_db, r_a, "s-", label="RZF (equal power, best U)")
    if any(x is not None for x in ex_a):
        y_ex = [x if x is not None else np.nan for x in ex_a]
        plt.plot(snr_grid_db, y_ex, "D--", label=f"Exhaustive P1 (U≤{args.exhaustive_Umax})")
    plt.xlabel(r"Normalized transmit power $10\log_{10}(P_{\max}/\sigma^2)$ (dB)")
    plt.ylabel("Balanced SINR (dB)")
    plt.title(f"(a) Small-scale fading only  |  K={args.K}, M={args.M}, σ²={args.sigma2:g}, samples={args.samples}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    ensure_dir(args.out_a); plt.savefig(args.out_a, dpi=220)

    # ---------------- Plot (b): large+small (includes exhaustive) -------
    t_b, z_b, r_b, ex_b = results["b"]
    plt.figure(figsize=(8.0, 5.2))
    plt.plot(snr_grid_db, t_b, "o-", label="Optimal P1 (max–min)")
    plt.plot(snr_grid_db, z_b, "^-", label="ZF (equal power, best U)")
    plt.plot(snr_grid_db, r_b, "s-", label="RZF (equal power, best U)")
    if any(x is not None for x in ex_b):
        y_ex = [x if x is not None else np.nan for x in ex_b]
        plt.plot(snr_grid_db, y_ex, "D--", label=f"Exhaustive P1 (U≤{args.exhaustive_Umax})")
    plt.xlabel(r"Normalized transmit power $10\log_{10}(P_{\max}/\sigma^2)$ (dB)")
    plt.ylabel("Balanced SINR (dB)")
    plt.title(f"(b) Large-scale + small-scale  |  fc={args.fc_GHz:g} GHz, K={args.K}, M={args.M}, samples={args.samples}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    ensure_dir(args.out_b); plt.savefig(args.out_b, dpi=220)

    if args.verbose:
        print(f"✅ Saved: {args.out_a} and {args.out_b}", flush=True)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
