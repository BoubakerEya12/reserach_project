# scripts/plot_fig5.py
# ---------------------------------------------------------------------
# Goal of this script
# ---------------------------------------------------------------------
# We want to reproduce two figures similar to Xia et al. Fig. 5:
#   (a) Only small-scale fading (no path loss / shadowing)
#   (b) Small-scale + large-scale fading using a 3GPP UMa path-loss model
#
# For each SNR point, we Monte-Carlo average the *balanced SINR*
# (i.e., the minimum user SINR among the scheduled users on the RB)
# across many random channel realizations, and we compare multiple methods:
#  - Optimal "teacher" (exhaustive max–min over user subsets with equal power)
#  - RZF with equal power
#  - ZF with equal power
#  - Exhaustive equal-power (baseline upper bound among equal-power schemes)
#  - WMMSE (sum-rate design) but evaluated with the min-SINR metric
# ---------------------------------------------------------------------

import os
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Channel utilities from our local package:
#   - gen_small_scale_iid_rayleigh: draw Rayleigh i.i.d. small-scale fading
#   - apply_large_scale           : multiply channels by sqrt(beta_u)
#   - noise_power_per_rb          : compute σ² = N0 * B_RB
# These are treated as building blocks to keep this script readable.
# ---------------------------------------------------------------------
from sim.channels import (
    gen_small_scale_iid_rayleigh,   # h: [K, N, M] ~ CN(0, I)
    apply_large_scale,              # beta -> h_eff
    noise_power_per_rb,             # sigma^2 = N0 * B_RB
)

# ---------------------------------------------------------------------
# 3GPP Urban Macro (UMa) large-scale model from our pathloss module.
# We try to import; if it fails, we fall back to a simple wrapper
# that returns something compatible (so the script doesn't crash).
# The boolean HAS_PATHLOSS lets us know if the detailed model is available.
# ---------------------------------------------------------------------
try:
    from sim.pathloss import gen_large_scale_3gpp_uma as UMAlarge
    from sim.pathloss import log_distance_shadowing as UMAwrapper  # simple fallback to UMa
    HAS_PATHLOSS = True
except Exception:
    HAS_PATHLOSS = False

# ---------------------------------------------------------------------
# Optimal teacher (P1): this function enumerates user subsets and
# maximizes the min-SINR under an equal-power constraint per RB,
# returning the best subset and beams. We reuse it as ground truth.
# ---------------------------------------------------------------------
from sim.exhaustive_teacher import best_group_by_balanced_sinr

# ---------------------------------------------------------------------
# Optional WMMSE baseline (optimizes weighted sum-rate).
# If the module is not present, we silently disable that curve.
# ---------------------------------------------------------------------
try:
    from sim.wmmse import wmmse_sumrate
    HAS_WMMSE = True
except Exception:
    HAS_WMMSE = False

# ============================ Small helpers ===========================

def lin2db(x):
    """
    Convert linear power/SINR to dB safely.

    Why np.maximum(x, 1e-12)?
      To avoid log10(0) or negative values due to numerical noise.
    """
    x = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(x, 1e-12))

def db2lin(db):
    """Convert dB to linear scale: 10^(db/10)."""
    return 10.0 ** (db / 10.0)

def clip_sinr(x_lin, max_db=60.0):
    """
    Some rare channel realizations can produce very large SINR and
    distort the average. We clip the *linear* SINR to a maximum
    (default 60 dB) to stabilize Monte-Carlo averages for panel (a).
    """
    return min(float(x_lin), 10.0**(max_db/10.0))

def draw_unitary_beams(M, U, rng):
    """
    Draw U random unit-norm beam directions in C^M.
    Steps:
      - draw complex Gaussian columns (M×U)
      - normalize each column to have Euclidean norm 1
    Returned type is complex64 to save memory / be consistent elsewhere.
    """
    re = rng.standard_normal(size=(M, U)).astype(np.float32)
    im = rng.standard_normal(size=(M, U)).astype(np.float32)
    W = (re + 1j*im) / np.sqrt(2.0)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    return W.astype(np.complex64)

def rzf_directions(H_sel, sigma2, P_rb):
    """
    Compute *directions* (unit-norm columns) of the RZF precoder.

    Inputs:
      H_sel : [U, M] user channels (rows are users)
      sigma2: noise power
      P_rb  : total power on the RB (used to pick the regularization)

    Internals:
      - We transpose to [M, U] because linear algebra is simpler that way.
      - Gram = H^H H is [U, U].
      - alpha = U * sigma2 / P_rb is a classic heuristic for RZF
        (more noise or more users => stronger regularization).
      - V = H (Gram + alpha I)^{-1} are the *directions* (not scaled by sqrt(p)).
      - We normalize columns to unit norm; power is applied later.
    """
    U, M = H_sel.shape
    H = H_sel.T  # [M,U]
    Gram = H.conj().T @ H
    alpha = (U * sigma2 / max(P_rb, 1e-12))   # classic heuristic
    A = Gram + alpha * np.eye(U, dtype=Gram.dtype)
    V = H @ np.linalg.pinv(A)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    return V.astype(np.complex64)

def build_equal_power_beams(H_sel, sigma2, P_rb, mode="rzf"):
    """
    Given the selected users H_sel [U, M], build beams with *equal power*:
      - if mode == "zf": use ZF directions: H (H^H H)^{-1}
      - else           : use RZF directions with heuristic alpha
      - then give each user the same power P_rb / U
      - return (V, p, W) where:
           V : [M, U] unit-norm directions
           p : [U]    per-user powers (equal)
           W : [M, U] actual precoder = V * sqrt(p)^T
    """
    U, M = H_sel.shape
    if mode == "zf":
        H = H_sel.T
        Gram = H.conj().T @ H
        V = H @ np.linalg.pinv(Gram)
        V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    else:
        V = rzf_directions(H_sel, sigma2, P_rb)
    p = np.full(U, P_rb / max(U, 1), dtype=np.float64)  # equal power
    W = V * np.sqrt(p.reshape(1, -1))
    return V.astype(np.complex64), p.astype(np.float64), W.astype(np.complex64)

def sinr_for_selection(HkM, sel, W_full, sigma2, I_out=None):
    """
    Compute user SINRs for a given *selection* and a *full* precoder.

    Inputs:
      HkM   : [K, M]  user channels (rows=users, columns=antennas)
      sel   : [U]     indices of scheduled users
      W_full: [M, K]  full precoder where non-selected columns are zeros
      sigma2: scalar  noise power
      I_out : [K] or None  optional out-of-cell interference per user

    Steps:
      - Compute HW = H W_full so that entry (u,u) is the useful signal
        and entries (u,sel\{u}) are the intra-RB interference.
      - For each scheduled user u, SINR_u = |HW[u,u]|^2 / (sum_{v!=u}|HW[u,v]|^2 + sigma2 + I_out[u])
      - Non-scheduled users keep SINR 0 (we don't use them).
    """
    K, M = HkM.shape
    sinr_u = np.zeros(K, dtype=np.float64)
    if sel.size == 0:
        return sinr_u
    HW = HkM @ W_full  # [K,K]
    for u in sel:
        sig = np.abs(HW[u, u])**2
        interf_intra = np.sum(np.abs(HW[u, sel])**2) - np.abs(HW[u, u])**2
        denom = interf_intra + sigma2
        if I_out is not None:
            denom += float(I_out[u])
        sinr_u[u] = sig / max(denom, 1e-30)  # avoid divide-by-zero
    return sinr_u

# =================== Toy inter-cell interference model ===================
def gen_out_of_cell_interference(K, M, U_int, P_int_per_cell, L_cells, rng):
    """
    Generate *average* out-of-cell interference per user.

    We simulate L_cells interfering base stations:
      - Each has random i.i.d. Rayleigh channels G [K, M] to our users.
      - Each uses U_int random unitary beams and equal power p_int.
      - For simplicity we don't include path loss; this is only to
        demonstrate robustness of the curves when interference exists.
    Output:
      I_out: [K] interference power per user (to be added to noise).
    """
    I_out = np.zeros(K, dtype=np.float64)
    for _ in range(L_cells):
        re = rng.standard_normal(size=(K, M)).astype(np.float32)
        im = rng.standard_normal(size=(K, M)).astype(np.float32)
        G = (re + 1j*im) / np.sqrt(2.0)  # [K,M]
        W_int = draw_unitary_beams(M, U_int, rng)  # [M,U_int]
        p_int = P_int_per_cell / max(U_int, 1)
        GW = G @ W_int  # [K, U_int]
        I_out += np.sum(np.abs(GW)**2, axis=1) * p_int
    return I_out

# ===================== Heuristics + exact evaluators =====================

def pick_subset_topU_by_norm(HkM, U_max):
    """
    Simple user selection heuristic:
      - Compute the channel norm of each user ‖h_u‖
      - Keep the top U_max users (or all if fewer than U_max)
      - Return sorted indices for reproducibility
    """
    norms = np.linalg.norm(HkM, axis=1)
    U = min(U_max, HkM.shape[0])
    sel = np.argsort(-norms)[:U]
    return np.sort(sel)

def zf_equal_power_value(HkM, sigma2, P_rb, U_max, I_out=None):
    """
    Evaluate the min-SINR (linear) achieved by:
      - selecting users by norm,
      - building ZF beams with equal power,
      - computing SINRs with optional interference.
    Also returns the full precoder and per-user powers for debugging/analysis.
    """
    sel = pick_subset_topU_by_norm(HkM, U_max)
    H_sel = HkM[sel, :]
    V, p, W = build_equal_power_beams(H_sel, sigma2, P_rb, mode="zf")
    K, M = HkM.shape
    W_full = np.zeros((M, K), dtype=np.complex64)
    p_full = np.zeros((K,), dtype=np.float64)
    for j, u in enumerate(sel):
        W_full[:, u] = W[:, j]; p_full[u] = p[j]
    sinr_u = sinr_for_selection(HkM, sel, W_full, sigma2, I_out=I_out)
    min_lin = float(np.min(sinr_u[sel]) if sel.size > 0 else 0.0)
    return sel, W_full, p_full, sinr_u.astype(np.float32), max(min_lin, 1e-30)

def rzf_equal_power_value(HkM, sigma2, P_rb, U_max, I_out=None):
    """
    Same as ZF, but using RZF directions with the heuristic regularization.
    """
    sel = pick_subset_topU_by_norm(HkM, U_max)
    H_sel = HkM[sel, :]
    V, p, W = build_equal_power_beams(H_sel, sigma2, P_rb, mode="rzf")
    K, M = HkM.shape
    W_full = np.zeros((M, K), dtype=np.complex64)
    p_full = np.zeros((K,), dtype=np.float64)
    for j, u in enumerate(sel):
        W_full[:, u] = W[:, j]; p_full[u] = p[j]
    sinr_u = sinr_for_selection(HkM, sel, W_full, sigma2, I_out=I_out)
    min_lin = float(np.min(sinr_u[sel]) if sel.size > 0 else 0.0)
    return sel, W_full, p_full, sinr_u.astype(np.float32), max(min_lin, 1e-30)

def exhaustive_equal_power_value(HkM, sigma2, P_rb, U_max, mode="rzf", I_out=None):
    """
    Upper-bound (among equal-power schemes):
      - Try all user subset sizes U = 1..min(U_max, K)
      - For each subset, build ZF/RZF equal-power beams
      - Keep the subset that maximizes min-SINR
    Complexity grows combinatorially with K (OK here for small K).
    """
    K, M = HkM.shape
    best = (-np.inf, None, None, None, None)  # (score, sel, W_full, p_full, sinr_u)
    users = list(range(K))
    for U in range(1, min(U_max, K) + 1):
        for comb in itertools.combinations(users, U):
            sel = np.array(comb, dtype=int)
            H_sel = HkM[sel, :]
            V, p, W = build_equal_power_beams(H_sel, sigma2, P_rb, mode=mode)
            W_full = np.zeros((M, K), dtype=np.complex64)
            p_full = np.zeros((K,), dtype=np.float64)
            for j, u in enumerate(sel):
                W_full[:, u] = W[:, j]; p_full[u] = p[j]
            sinr_u = sinr_for_selection(HkM, sel, W_full, sigma2, I_out=I_out)
            min_lin = float(np.min(sinr_u[sel]))
            if min_lin > best[0]:
                best = (min_lin, sel.copy(), W_full.copy(), p_full.copy(), sinr_u.copy())
    if best[1] is None:
        # Degenerate fallback if K==0 or something odd
        return np.array([], dtype=int), np.zeros((M, K), dtype=np.complex64), np.zeros((K,)), np.zeros((K,), dtype=np.float32), 1e-30
    min_lin, sel, W_full, p_full, sinr_u = best
    return sel, W_full, p_full, sinr_u.astype(np.float32), max(min_lin, 1e-30)

def wmmse_value(HkM, sigma2, P_rb, U_max, I_out=None):
    """
    Use WMMSE (sum-rate) to design a precoder, but *evaluate*
    with the min-SINR objective (so we can compare apples to apples).

    If the WMMSE module is not available, return a tiny value so the curve
    won't be plotted (the caller checks availability).
    """
    if not HAS_WMMSE:
        return 1e-30
    sel = pick_subset_topU_by_norm(HkM, U_max)
    U = sel.size
    if U == 0:
        return 1e-30
    H_sel = HkM[sel, :].T  # [M,U] for the WMMSE function’s convention
    W, p_u, sinr_u = wmmse_sumrate(H=H_sel, sigma2=sigma2 + 0.0, Pmax=P_rb, iters=50)
    # Map to full-K and recompute SINRs (with optional interference)
    M = H_sel.shape[0]
    W_full = np.zeros((M, HkM.shape[0]), dtype=np.complex64)
    for j, u in enumerate(sel):
        W_full[:, u] = W[:, j]
    sinr_full = sinr_for_selection(HkM, sel, W_full, sigma2, I_out=I_out)
    return float(max(np.min(sinr_full[sel]), 1e-30))

# ========================== One-sample evaluators ==========================

def compute_methods_small_only(HkM, sigma2, P_rb, U_max, I_out=None, with_wmmse=False):
    """
    Evaluate all methods on a *single* small-scale channel draw:
      - Optimal teacher (P1)
      - ZF, RZF (equal power)
      - Exhaustive equal-power
      - (optional) WMMSE
    Return their *linear* min-SINR scores, clipped for panel (a).
    """
    # Optimal teacher (P1) expects Hn = [M,K]
    sel_o, W_o, p_o, sinr_o, score_lin0 = best_group_by_balanced_sinr(
        Hn=HkM.T, sigma2=sigma2, P_rb=P_rb, U_max=U_max
    )
    # If we also add out-of-cell interference, recompute min-SINR on top of teacher beams
    if I_out is not None and sel_o is not None and sel_o.size > 0:
        sinr_o2 = sinr_for_selection(HkM, sel_o, W_o, sigma2, I_out=I_out)
        opt_lin = float(max(np.min(sinr_o2[sel_o]), 1e-30))
    else:
        opt_lin = float(max(score_lin0, 1e-30))

    # Heuristics + exhaustive (equal power)
    _, _, _, _, zf_lin  = zf_equal_power_value (HkM, sigma2, P_rb, U_max, I_out=I_out)
    _, _, _, _, rzf_lin = rzf_equal_power_value(HkM, sigma2, P_rb, U_max, I_out=I_out)
    _, _, _, _, ex_lin  = exhaustive_equal_power_value(HkM, sigma2, P_rb, U_max, mode="rzf", I_out=I_out)

    wm_lin = None
    if with_wmmse:
        wm_lin = wmmse_value(HkM, sigma2, P_rb, U_max, I_out=I_out)

    # Clip outliers only for panel (a) to stabilize the average
    opt_lin = clip_sinr(opt_lin); zf_lin = clip_sinr(zf_lin)
    rzf_lin = clip_sinr(rzf_lin); ex_lin = clip_sinr(ex_lin)
    if wm_lin is not None: wm_lin = clip_sinr(wm_lin)

    return opt_lin, zf_lin, rzf_lin, ex_lin, wm_lin

def compute_methods_large_plus_small(HkM_eff, sigma2, P_rb, U_max, I_out=None, with_wmmse=False):
    """
    Same as above, but now HkM_eff already includes the large-scale gain sqrt(beta_u):
    we directly call the teacher and heuristics on the effective channels.
    """
    sel_o, W_o, p_o, sinr_o, score_lin0 = best_group_by_balanced_sinr(
        Hn=HkM_eff.T, sigma2=sigma2, P_rb=P_rb, U_max=U_max
    )
    if I_out is not None and sel_o is not None and sel_o.size > 0:
        sinr_o2 = sinr_for_selection(HkM_eff, sel_o, W_o, sigma2, I_out=I_out)
        opt_lin = float(max(np.min(sinr_o2[sel_o]), 1e-30))
    else:
        opt_lin = float(max(score_lin0, 1e-30))

    _, _, _, _, zf_lin  = zf_equal_power_value (HkM_eff, sigma2, P_rb, U_max, I_out=I_out)
    _, _, _, _, rzf_lin = rzf_equal_power_value(HkM_eff, sigma2, P_rb, U_max, I_out=I_out)
    _, _, _, _, ex_lin  = exhaustive_equal_power_value(HkM_eff, sigma2, P_rb, U_max, mode="rzf", I_out=I_out)

    wm_lin = None
    if with_wmmse:
        wm_lin = wmmse_value(HkM_eff, sigma2, P_rb, U_max, I_out=I_out)

    # No clipping here by default (panel b shows the real scale drop)
    return opt_lin, zf_lin, rzf_lin, ex_lin, wm_lin

# ============================== Monte-Carlo ===============================

def run_case(K, M, N, sigma2, samples, snr_grid_db, include_large, rng, verbose, U_max,
             n_interf_cells=0, U_int=1, Pint_rel_db=0.0,
             fc_GHz=3.5, h_bs=25.0, h_ut=1.5, los_mode="prob", cell_radius_m=250.0,
             with_wmmse=False):
    """
    Outer Monte-Carlo loop over SNR points.
    For each SNR value:
      - compute P_rb from SNR:  P_rb = sigma2 * 10^(SNR_dB/10)
      - repeat 'samples' times:
          (a) draw small-scale fading and evaluate methods
          (b) if include_large: draw beta_u via 3GPP UMa and evaluate again
      - average the *linear* min-SINR per method, then convert to dB for plotting.
    """
    # We store per-curve y-values across SNR grid for panels (a) and (b)
    ys_a_opt, ys_a_zf, ys_a_rzf, ys_a_ex, ys_a_wm = [], [], [], [], []
    ys_b_opt, ys_b_zf, ys_b_rzf, ys_b_ex, ys_b_wm = [], [], [], [], []

    for snr_db in snr_grid_db:
        P_rb = sigma2 * db2lin(snr_db)  # per-RB total power for this SNR

        # Accumulators (linear) + counters for averaging
        acc_a = {'opt': 0.0, 'zf': 0.0, 'rzf': 0.0, 'ex': 0.0, 'wm': 0.0}
        cnt_a = {'opt': 0,    'zf': 0,    'rzf': 0,    'ex': 0,    'wm': 0}

        acc_b = {'opt': 0.0, 'zf': 0.0, 'rzf': 0.0, 'ex': 0.0, 'wm': 0.0}
        cnt_b = {'opt': 0,    'zf': 0,    'rzf': 0,    'ex': 0,    'wm': 0}

        if verbose:
            print(f"[SNR={snr_db:>2} dB] sample 0/{samples}", flush=True)

        for s in range(samples):
            if verbose and (s % max(1, samples // 5) == 0) and s > 0:
                print(f"[SNR={snr_db:>2} dB] sample {s}/{samples}", flush=True)

            # ---- (a) Small-scale Rayleigh only (no path loss) ----
            # gen_small_scale_iid_rayleigh returns h: [K, N, M]
            h_ss = gen_small_scale_iid_rayleigh(M, K, N, rng)
            # we only need one RB realization to evaluate the methods
            HkM = h_ss[:, 0, :]  # [K, M]

            # Optional: add toy out-of-cell interference
            I_out = None
            if n_interf_cells > 0:
                P_int_cell = P_rb * db2lin(Pint_rel_db)  # relative power wrt desired cell
                I_out = gen_out_of_cell_interference(K, M, U_int, P_int_cell, n_interf_cells, rng)

            opt_lin, zf_lin, rzf_lin, ex_lin, wm_lin = compute_methods_small_only(
                HkM, sigma2, P_rb, U_max, I_out=I_out, with_wmmse=with_wmmse
            )
            # Accumulate and count to compute averages later
            acc_a['opt'] += opt_lin; cnt_a['opt'] += 1
            acc_a['zf']  += zf_lin;  cnt_a['zf']  += 1
            acc_a['rzf'] += rzf_lin; cnt_a['rzf'] += 1
            acc_a['ex']  += ex_lin;  cnt_a['ex']  += 1
            if wm_lin is not None:
                acc_a['wm'] += wm_lin; cnt_a['wm'] += 1

            # ---- (b) Add large-scale (3GPP UMa) on top of small-scale ----
            if include_large:
                if HAS_PATHLOSS:
                    # Detailed UMa model (random LOS/NLOS + shadowing)
                    beta, is_los, _ = UMAlarge(
                        K=K, fc_GHz=fc_GHz, h_bs=h_bs, h_ut=h_ut,
                        los_mode=los_mode, cell_radius_m=cell_radius_m, rng=rng
                    )
                else:
                    # Fallback wrapper (still consistent with UMa)
                    beta, _ = UMAwrapper(rng=rng, K=K, fc_GHz=fc_GHz)

                # Apply beta_u to each user's small-scale: h_eff[u,:,:] = sqrt(beta_u) * h_ss[u,:,:]
                h_eff = apply_large_scale(h_ss, beta)   # [K, N, M]
                HkM_eff = h_eff[:, 0, :]               # [K, M] for RB 0

                I_out_b = None
                if n_interf_cells > 0:
                    P_int_cell = P_rb * db2lin(Pint_rel_db)
                    I_out_b = gen_out_of_cell_interference(K, M, U_int, P_int_cell, n_interf_cells, rng)

                opt_lin_b, zf_lin_b, rzf_lin_b, ex_lin_b, wm_lin_b = compute_methods_large_plus_small(
                    HkM_eff, sigma2, P_rb, U_max, I_out=I_out_b, with_wmmse=with_wmmse
                )
                acc_b['opt'] += opt_lin_b; cnt_b['opt'] += 1
                acc_b['zf']  += zf_lin_b;  cnt_b['zf']  += 1
                acc_b['rzf'] += rzf_lin_b; cnt_b['rzf'] += 1
                acc_b['ex']  += ex_lin_b;  cnt_b['ex'] += 1
                if wm_lin_b is not None:
                    acc_b['wm'] += wm_lin_b; cnt_b['wm'] += 1

        # After all samples at this SNR, convert averaged *linear* min-SINR to dB
        ys_a_opt.append(lin2db(acc_a['opt'] / max(1, cnt_a['opt'])))
        ys_a_zf .append(lin2db(acc_a['zf']  / max(1, cnt_a['zf'])))
        ys_a_rzf.append(lin2db(acc_a['rzf'] / max(1, cnt_a['rzf'])))
        ys_a_ex .append(lin2db(acc_a['ex']  / max(1, cnt_a['ex'])))
        if cnt_a['wm'] > 0:
            ys_a_wm.append(lin2db(acc_a['wm'] / cnt_a['wm']))

        if include_large:
            ys_b_opt.append(lin2db(acc_b['opt'] / max(1, cnt_b['opt'])))
            ys_b_zf .append(lin2db(acc_b['zf']  / max(1, cnt_b['zf'])))
            ys_b_rzf.append(lin2db(acc_b['rzf'] / max(1, cnt_b['rzf'])))
            ys_b_ex .append(lin2db(acc_b['ex']  / max(1, cnt_b['ex'])))
            if cnt_b['wm'] > 0:
                ys_b_wm.append(lin2db(acc_b['wm'] / cnt_b['wm']))

    # Return panel (a) arrays and, if requested, panel (b) arrays.
    return (np.array(ys_a_opt), np.array(ys_a_zf), np.array(ys_a_rzf), np.array(ys_a_ex),
            np.array(ys_a_wm) if len(ys_a_wm)>0 else None), \
           (np.array(ys_b_opt), np.array(ys_b_zf), np.array(ys_b_rzf), np.array(ys_b_ex),
            np.array(ys_b_wm) if (include_large and len(ys_b_wm)>0) else None) \
           if include_large else (None, None, None, None, None)

# ================================ Plotting ================================

def make_figures(snr_grid_db, panel_a, panel_b, K, M, samples, include_large, with_wmmse):
    """
    Create the matplotlib figures for panels (a) and (b).
    We pass the y-values for each curve and add labels, titles and grids.
    """
    # Panel (a)
    fig_a, ax = plt.subplots(figsize=(9, 5.5))
    y_opt, y_zf, y_rzf, y_ex, y_wm = panel_a
    ax.plot(snr_grid_db, y_opt, 'o-', label='Optimal (max–min)')
    ax.plot(snr_grid_db, y_rzf, 's-', label='RZF (equal power)')
    ax.plot(snr_grid_db, y_zf,  '^-', label='ZF (equal power)')
    ax.plot(snr_grid_db, y_ex,  'd-', label=f'Exhaustive (U≤{K})')
    if with_wmmse and (y_wm is not None):
        ax.plot(snr_grid_db, y_wm, 'x-', label='WMMSE (sum-rate)')
    ax.set_title(f"(a) Small-scale fading only  |  K={K}, M={M}, samples={samples}")
    ax.set_xlabel(r"Normalized transmit power $10\log_{10}(P_{\max}/\sigma^2)$ (dB)")
    ax.set_ylabel("Balanced SINR (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    # Panel (b) only if requested and data is present
    fig_b = None
    if include_large and (panel_b[0] is not None):
        fig_b, ax2 = plt.subplots(figsize=(9, 5.5))
        y_opt_b, y_zf_b, y_rzf_b, y_ex_b, y_wm_b = panel_b
        ax2.plot(snr_grid_db, y_opt_b, 'o-', label='Optimal (max–min)')
        ax2.plot(snr_grid_db, y_rzf_b, 's-', label='RZF (equal power)')
        ax2.plot(snr_grid_db, y_zf_b,  '^-', label='ZF (equal power)')
        ax2.plot(snr_grid_db, y_ex_b,  'd-', label=f'Exhaustive (U≤{K})')
        if with_wmmse and (y_wm_b is not None):
            ax2.plot(snr_grid_db, y_wm_b, 'x-', label='WMMSE (sum-rate)')
        ax2.set_title(f"(b) Large-scale + small-scale (3GPP UMa)  |  K={K}, M={M}, samples={samples}")
        ax2.set_xlabel(r"Normalized transmit power $10\log_{10}(P_{\max}/\sigma^2)$ (dB)")
        ax2.set_ylabel("Balanced SINR (dB)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right')

    return fig_a, fig_b

# ================================== CLI ==================================

def main():
    """
    Command-line interface:
      - choose sizes (N,K,M), number of samples, seed, etc.
      - choose whether to include large-scale (panel b) and WMMSE curve
      - choose 3GPP UMa parameters for the large-scale model
      - save figures to ./figs if --save is given
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=6)         # number of RBs to draw small-scale for (we only use RB 0)
    ap.add_argument("--K", type=int, default=4)         # number of users
    ap.add_argument("--M", type=int, default=6)         # number of BS antennas
    ap.add_argument("--Umax", type=int, default=None)   # max users per RB; default=min(M,K)
    ap.add_argument("--samples", type=int, default=2000)# Monte-Carlo samples per SNR
    ap.add_argument("--seed", type=int, default=1234)   # RNG seed for reproducibility
    ap.add_argument("--save", action="store_true")      # save figures to disk
    ap.add_argument("--no-show", action="store_true")   # run headless (useful on servers)
    ap.add_argument("--verbose", action="store_true")   # print progress
    ap.add_argument("--N0", type=float, default=1e-20)  # noise PSD (W/Hz)
    ap.add_argument("--BRB", type=float, default=180e3) # RB bandwidth (Hz)
    ap.add_argument("--with-large", action="store_true")# also produce panel (b)
    ap.add_argument("--with-wmmse", action="store_true", help="Add WMMSE curve (if sim.wmmse is available).")

    # Inter-cell interference controls (optional)
    ap.add_argument("--n-interf-cells", type=int, default=0)  # number of interfering cells
    ap.add_argument("--Uint", type=int, default=1)            # beams/users per interfering cell
    ap.add_argument("--Pint-rel-db", type=float, default=0.0) # interference power relative to desired cell (dB)

    # 3GPP UMa parameters for panel (b)
    ap.add_argument("--fc-GHz", type=float, default=3.5)
    ap.add_argument("--h-bs", type=float, default=25.0)
    ap.add_argument("--h-ut", type=float, default=1.5)
    ap.add_argument("--los-mode", choices=["prob","always_los","always_nlos"], default="prob")
    ap.add_argument("--cell-radius-m", type=float, default=250.0)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # Resolve dimensions and defaults
    K, M, N = args.K, args.M, args.N
    U_max = args.Umax if args.Umax is not None else min(M, K)

    # Compute noise power per RB: sigma² = N0 * B_RB
    sigma2 = float(noise_power_per_rb(args.N0, args.BRB))
    print(f"[noise] sigma² = {sigma2:.3e} W")

    # SNR grid in dB along the x-axis (0, 3, 6, …, 30)
    snr_grid_db = list(range(0, 31, 3))

    # Run Monte-Carlo across the SNR grid; panel_b is None if with_large is False
    panel_a, panel_b = run_case(
        K=K, M=M, N=N, sigma2=sigma2, samples=args.samples, snr_grid_db=snr_grid_db,
        include_large=args.with_large, rng=rng, verbose=args.verbose, U_max=U_max,
        n_interf_cells=args.n_interf_cells, U_int=args.Uint, Pint_rel_db=args.Pint_rel_db,
        fc_GHz=args.fc_GHz, h_bs=args.h_bs, h_ut=args.h_ut,
        los_mode=args.los_mode, cell_radius_m=args.cell_radius_m,
        with_wmmse=args.with_wmmse
    )

    # Build the matplotlib figures
    fig_a, fig_b = make_figures(snr_grid_db, panel_a, panel_b, K, M, args.samples, args.with_large, args.with_wmmse)

    # Optionally save to disk; panel (b) is saved only if it exists
    if args.save:
        os.makedirs("figs", exist_ok=True)
        fig_a.savefig("figs/fig5_a.png", dpi=200, bbox_inches="tight")
        if fig_b is not None:
            fig_b.savefig("figs/fig5_b.png", dpi=200, bbox_inches="tight")
        print("Saved: figs/fig5_a.png")
        if fig_b is not None:
            print("Saved: figs/fig5_b.png")

    # Show interactively unless --no-show was given
    if not args.no_show:
        plt.show()
    else:
        plt.close('all')

# Standard Python main guard so the script can be imported without running.
if __name__ == "__main__":
    main()
