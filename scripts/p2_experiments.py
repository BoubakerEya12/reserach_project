# scripts/p2_experiments.py
# ---------------------------------------------------------
# P2 : Power minimization under SINR / rate constraints
#
# - Single-cell downlink, Nt antennas, K users, NRB RBs
# - Per-RB ZF beamforming
# - Each user must be scheduled on exactly 1 RB
# - At most Umax users per RB
# - P2: minimize sum power subject to SINR >= gamma_target
#
# We generate several figures:
#   Fig1: Power per RB (bar plot) for one schedule
#   Fig2: Total power vs SINR target (average over many channels)
#   Fig3: Total power vs number of users K
#
# Run from terminal:
#   python -m scripts.p2_experiments
# ---------------------------------------------------------

import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
import pulp


# ---------- Small utilities ----------

def db2lin(x_db: float) -> float:
    return 10.0 ** (x_db / 10.0)


def lin2db(x_lin: float, eps: float = 1e-15) -> float:
    x_lin = max(x_lin, eps)
    return 10.0 * np.log10(x_lin)


def to_complex_gaussian(shape, rng):
    """CN(0,1) i.i.d."""
    re = rng.normal(0.0, 1.0 / np.sqrt(2.0), size=shape)
    im = rng.normal(0.0, 1.0 / np.sqrt(2.0), size=shape)
    return re + 1j * im


def unit_norm_rows(V, eps: float = 1e-12):
    """
    Normalise chaque ligne de V à norme 2 unitaire.
    V : [U, Nt] (complex or real)
    """
    V = np.asarray(V)
    norms = np.sqrt(np.maximum(np.sum(np.abs(V) ** 2, axis=1, keepdims=True), eps))
    return V / norms


# ---------- Channel + path-loss model (cohérent avec ton LaTeX) ----------

def generate_large_scale(K, rng,
                         d_min=50.0, d_max=250.0,
                         PL0=30.0, d0=1.0, alpha=3.5,
                         sigma_sh=8.0):
    """
    Génère les gains de grande échelle β_u selon ton modèle:

      β_u = 10^{-(PL0 + 10α log10(d_u/d0) + Xσ)/10}

    Retourne:
      betas: [K] (linear)
      dists: [K] (m)
    """
    dists = rng.uniform(d_min, d_max, size=K)
    X_sigma = rng.normal(0.0, sigma_sh, size=K)  # dB
    PL_dB = PL0 + 10.0 * alpha * np.log10(dists / d0) + X_sigma
    betas = 10.0 ** (-PL_dB / 10.0)
    return betas, dists


def draw_effective_channels_per_RB(K, Nt, NRB, betas, rng):
    """
    Pour chaque RB, génère un canal effectif H_n de taille [K, Nt].
    Ici, on stocke liste Hn_list[n] = [K, Nt] (tous les users disponibles),
    mais le scheduling choisira un sous-ensemble.
    """
    Hn_list = []
    for _ in range(NRB):
        # small-scale Rayleigh CN(0,I)
        H_small = to_complex_gaussian((K, Nt), rng)
        # effective = sqrt(beta_u) * h
        H_eff = (np.sqrt(betas)[:, None]) * H_small
        Hn_list.append(H_eff)
    return Hn_list


# ---------- ZF beamforming + required power ----------

def zf_beams(H):
    """
    Classical zero-forcing beamformer for MU-MISO.

    H : array of shape [U, Nt] (rows = users, columns = BS antennas)

    Returns
    -------
    V : array [U, Nt]
        Row u is the (unit-norm) ZF beam for user u.
    gains : array [U]
        Effective channel gains |h_u^H v_u|^2 for each user.
    """
    H = np.asarray(H)

    # Hermitian: conjugate transpose
    Hc = H.conj().T          # shape [Nt, U]

    # ZF: V = H^H (H H^H)^{-1}
    A = H @ Hc               # [U, U]
    Vt = Hc @ np.linalg.pinv(A)  # [Nt, U]

    # Put beams as rows [U, Nt]
    V = Vt.T                 # [U, Nt]

    # Normalize each row to unit norm
    V = unit_norm_rows(V)

    # Effective gain for each user: h_u^H v_u
    hv = np.sum(H * V.conj(), axis=1)
    gains = np.abs(hv) ** 2

    return V, gains


def required_power_for_subset(H_sub, gamma_lin, sigma2, P_rb_max):
    """
    H_sub: [U_sub, Nt]
    gamma_lin: scalar or [U_sub]
    sigma2: noise power on RB
    Retourne:
       P_rb (float), p_users (array length U_sub), feasible (bool)
    """
    U_sub, _ = H_sub.shape
    if U_sub == 0:
        return 0.0, np.zeros(0), True

    if np.isscalar(gamma_lin):
        gam = np.full(U_sub, gamma_lin, dtype=np.float64)
    else:
        gam = np.array(gamma_lin, dtype=np.float64)

    V, gains = zf_beams(H_sub)

    # Si un gain est trop petit, la puissance explose → on déclare infeasible
    if np.any(gains < 1e-12):
        return 0.0, np.zeros(U_sub), False

    # SINR_u = p_u * g_u / sigma2   (ZF ⇒ pas d'interférence multi-utilisateur)
    # p_u >= gamma_u * sigma2 / g_u
    p_users = gam * sigma2 / gains
    P_rb = float(np.sum(p_users))

    if P_rb > P_rb_max:
        return P_rb, p_users, False
    return P_rb, p_users, True


# ---------- Construire les coûts (pré-calcul) ----------

def build_rb_candidates(Hn_list, gamma_lin, sigma2, Umax, P_rb_max):
    """
    Pour chaque RB n, on considère tous les sous-ensembles de users de taille 1..Umax.
    Pour chaque subset S, on calcule:
       - puissance minimale P_rb(S)
       - vecteur de puissance p_u(S)
    On ne garde que les subsets faisables.

    Retourne:
       candidates_per_rb: list de longueur NRB
         où candidates_per_rb[n] = list de dicts:
           {
             "users": tuple(indices_users),
             "P_rb": puissance totale,
             "p_users": array [len(S)]
           }
    """
    NRB = len(Hn_list)
    K, Nt = Hn_list[0].shape
    candidates_per_rb = []

    for n in range(NRB):
        Hn = Hn_list[n]  # [K, Nt]
        cand_list = []
        for usize in range(1, Umax + 1):
            for subset in itertools.combinations(range(K), usize):
                idx = np.array(subset, dtype=int)
                H_sub = Hn[idx, :]
                P_rb, p_users, feasible = required_power_for_subset(
                    H_sub, gamma_lin, sigma2, P_rb_max
                )
                if feasible:
                    cand_list.append({
                        "users": subset,
                        "P_rb": P_rb,
                        "p_users": p_users,
                    })
        candidates_per_rb.append(cand_list)
    return candidates_per_rb


# ---------- ILP global : choisir un subset par RB pour minimiser la somme de puissance ----------

def solve_global_schedule(candidates_per_rb, K, NRB):
    """
    MILP:
       Variables y_{n,s} in {0,1} : subset s choisi sur RB n.
       Contraintes :
         - pour chaque RB n: sum_s y_{n,s} <= 1
         - pour chaque user u: sum_{n,s: u in S_{n,s}} y_{n,s} == 1
           (chaque user doit être planifié sur exactement 1 RB)
       Objectif : min sum_{n,s} P_rb(n,s) * y_{n,s}
    """
    prob = pulp.LpProblem("P2_min_power_schedule", pulp.LpMinimize)

    y_vars = {}
    for n in range(NRB):
        for s_idx, cand in enumerate(candidates_per_rb[n]):
            y_vars[(n, s_idx)] = pulp.LpVariable(
                f"y_{n}_{s_idx}", lowBound=0, upBound=1, cat=pulp.LpBinary
            )

    # Si aucun candidat n'existe, on ne peut rien faire
    if len(y_vars) == 0:
        return [None] * NRB, 0.0

    # Objective
    prob += pulp.lpSum(
        cand["P_rb"] * y_vars[(n, s_idx)]
        for n in range(NRB)
        for s_idx, cand in enumerate(candidates_per_rb[n])
    )

    # Constraints: at most 1 subset per RB
    for n in range(NRB):
        prob += pulp.lpSum(
            y_vars[(n, s_idx)] for s_idx, _ in enumerate(candidates_per_rb[n])
        ) <= 1, f"RB_{n}_at_most_one_subset"

    # Constraints: each user scheduled on EXACTLY one RB
    for u in range(K):
        prob += pulp.lpSum(
            y_vars[(n, s_idx)]
            for n in range(NRB)
            for s_idx, cand in enumerate(candidates_per_rb[n])
            if u in cand["users"]
        ) == 1, f"user_{u}_exactly_one_RB"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract schedule
    schedule = [None] * NRB
    total_power = 0.0
    for n in range(NRB):
        chosen = None
        best_p = None
        for s_idx, cand in enumerate(candidates_per_rb[n]):
            if pulp.value(y_vars[(n, s_idx)]) > 0.5:
                chosen = cand
                best_p = cand["P_rb"]
                break
        if chosen is not None:
            schedule[n] = chosen
            total_power += best_p
        else:
            schedule[n] = None

    return schedule, total_power


# ---------- Expérience de base : une réalisation de canal + figure 1 ----------

def run_single_instance_fig1(seed=0):
    """
    1) On génère un scénario fixe
    2) On résout P2 (min-power)
    3) On trace bar plot de puissance par RB
    """
    rng = np.random.default_rng(seed)

    Nt = 4
    K = 6
    NRB = 3
    Umax = 3

    BRB = 180e3  # 180 kHz
    N0_dBm_per_Hz = -174.0
    N0_W_per_Hz = 10.0 ** ((N0_dBm_per_Hz - 30.0) / 10.0)
    sigma2 = N0_W_per_Hz * BRB  # noise power per RB

    Ptot = 10.0  # W
    P_rb_max = Ptot  # ici on ne contraint pas par RB de façon stricte

    gamma_dB = 10.0
    gamma_lin = db2lin(gamma_dB)

    betas, dists = generate_large_scale(K, rng)
    Hn_list = draw_effective_channels_per_RB(K, Nt, NRB, betas, rng)

    candidates_per_rb = build_rb_candidates(
        Hn_list, gamma_lin, sigma2, Umax, P_rb_max
    )
    schedule, total_power = solve_global_schedule(
        candidates_per_rb, K, NRB
    )

    print(f"[FIG1] Total power = {total_power:.4e} W (gamma = {gamma_dB} dB)")

    # Power per RB (0 for RB not used)
    rb_powers = []
    for n in range(NRB):
        if schedule[n] is None:
            rb_powers.append(0.0)
        else:
            rb_powers.append(schedule[n]["P_rb"])

    # Save CSV for Fig1
    rb_indices = np.arange(NRB)
    data_fig1 = np.column_stack([rb_indices, rb_powers])
    np.savetxt(
        "results/p2_fig1_rb_powers.csv",
        data_fig1,
        delimiter=",",
        header="RB_index,Power_W",
        comments="",
    )

    # Figure 1
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(NRB), rb_powers)
    plt.xlabel("RB index")
    plt.ylabel("Power per RB (W)")
    plt.title("P2 — Power per RB for chosen schedule")
    plt.tight_layout()
    plt.savefig("results/p2_fig1_rb_powers.png", dpi=300)
    plt.show()

    return total_power


# ---------- Expérience 2 : Total power vs SINR target ----------

def experiment_fig2_total_power_vs_sinr(seed=1):
    rng = np.random.default_rng(seed)

    Nt = 4
    K = 6
    NRB = 3
    Umax = 3

    BRB = 180e3
    N0_dBm_per_Hz = -174.0
    N0_W_per_Hz = 10.0 ** ((N0_dBm_per_Hz - 30.0) / 10.0)
    sigma2 = N0_W_per_Hz * BRB

    Ptot = 10.0
    P_rb_max = Ptot

    gamma_dB_list = [0.0, 5.0, 10.0, 15.0, 20.0]
    avg_powers = []

    num_channels = 50  # nb de réalisations pour la moyenne

    for gamma_dB in gamma_dB_list:
        gamma_lin = db2lin(gamma_dB)
        powers = []

        for _ in range(num_channels):
            betas, dists = generate_large_scale(K, rng)
            Hn_list = draw_effective_channels_per_RB(K, Nt, NRB, betas, rng)
            candidates_per_rb = build_rb_candidates(
                Hn_list, gamma_lin, sigma2, Umax, P_rb_max
            )
            schedule, total_power = solve_global_schedule(
                candidates_per_rb, K, NRB
            )
            powers.append(total_power)

        avg_powers.append(np.mean(powers))
        print(f"[FIG2] gamma={gamma_dB:.1f} dB → avg total power={avg_powers[-1]:.4e} W")

    # Save CSV for Fig2
    data_fig2 = np.column_stack([gamma_dB_list, avg_powers])
    np.savetxt(
        "results/p2_fig2_total_power_vs_sinr.csv",
        data_fig2,
        delimiter=",",
        header="gamma_dB,avg_total_power_W",
        comments="",
    )

    plt.figure(figsize=(6, 4))
    plt.plot(gamma_dB_list, avg_powers, marker="o")
    plt.xlabel("Target SINR (dB)")
    plt.ylabel("Average total power (W)")
    plt.title("P2 — Total power vs SINR target")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/p2_fig2_total_power_vs_sinr.png", dpi=300)
    plt.show()


# ---------- Expérience 3 : Total power vs nombre de users K ----------

def experiment_fig3_total_power_vs_K(seed=2):
    rng = np.random.default_rng(seed)

    Nt = 4
    NRB = 3
    Umax = 3

    BRB = 180e3
    N0_dBm_per_Hz = -174.0
    N0_W_per_Hz = 10.0 ** ((N0_dBm_per_Hz - 30.0) / 10.0)
    sigma2 = N0_W_per_Hz * BRB

    Ptot = 10.0
    P_rb_max = Ptot

    gamma_dB = 10.0
    gamma_lin = db2lin(gamma_dB)

    K_list = [3, 4, 5, 6, 7, 8]
    avg_powers = []

    num_channels = 50

    for K in K_list:
        powers = []
        for _ in range(num_channels):
            betas, dists = generate_large_scale(K, rng)
            Hn_list = draw_effective_channels_per_RB(K, Nt, NRB, betas, rng)
            candidates_per_rb = build_rb_candidates(
                Hn_list, gamma_lin, sigma2, Umax, P_rb_max
            )
            schedule, total_power = solve_global_schedule(
                candidates_per_rb, K, NRB
            )
            powers.append(total_power)
        avg_powers.append(np.mean(powers))
        print(f"[FIG3] K={K} → avg total power={avg_powers[-1]:.4e} W")

    # Save CSV for Fig3
    data_fig3 = np.column_stack([K_list, avg_powers])
    np.savetxt(
        "results/p2_fig3_total_power_vs_K.csv",
        data_fig3,
        delimiter=",",
        header="K,avg_total_power_W",
        comments="",
    )

    plt.figure(figsize=(6, 4))
    plt.plot(K_list, avg_powers, marker="o")
    plt.xlabel("Number of users K")
    plt.ylabel("Average total power (W)")
    plt.title(f"P2 — Total power vs K (gamma={gamma_dB} dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/p2_fig3_total_power_vs_K.png", dpi=300)
    plt.show()


# ---------- Main ----------

def main():
    os.makedirs("results", exist_ok=True)

    # Fig1: single realization, power per RB
    run_single_instance_fig1(seed=0)

    # Fig2: average total power vs SINR
    experiment_fig2_total_power_vs_sinr(seed=1)

    # Fig3: average total power vs K
    experiment_fig3_total_power_vs_K(seed=2)


if __name__ == "__main__":
    main()
