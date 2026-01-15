# scripts/p2_full.py
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

try:
    import pulp  # for ILP
except ImportError:
    raise SystemExit("Please install pulp: pip install pulp")

from sim.p2_solver import solve_min_power_sinr
from sim.p2_solver import solve_p2_given_schedule

# --- Try to use your project channel generator ---
try:
    # In your project this likely has signature: draw_small_scale(K, Nt, rng)
    from sim.channels import draw_small_scale  # your util
    HAS_PROJECT_CHANNEL = True
except Exception:
    HAS_PROJECT_CHANNEL = False

    def draw_small_scale(K, Nt, rng=None):
        """Fallback: [K, Nt] i.i.d. CN(0,1) Rayleigh."""
        if rng is None:
            rng = np.random.default_rng()
        Hr = (rng.standard_normal((K, Nt)) +
              1j * rng.standard_normal((K, Nt))) / np.sqrt(2.0)
        return Hr


def effective_channels_per_RB(K, Nt, NRB, betas=None, rng=None):
    """
    Returns list of per-RB effective channels [K,Nt]:
    H_eff = sqrt(beta_u) * H (per user).

    Handles both:
      - draw_small_scale(K, Nt, rng)
      - draw_small_scale(K, Nt)
    """
    if betas is None:
        betas = np.ones(K)

    if rng is None:
        rng = np.random.default_rng(1234)

    Hn_list = []
    for _ in range(NRB):
        # Try with rng argument first (your project style)
        try:
            H = draw_small_scale(K, Nt, rng)
        except TypeError:
            # Fallback if draw_small_scale only takes (K, Nt)
            H = draw_small_scale(K, Nt)

        H_eff = (np.sqrt(betas).reshape(-1, 1)) * H  # per-user large-scale
        Hn_list.append(H_eff)
    return Hn_list


def all_subsets_users(K, Umax):
    """
    Generate all subsets of users with size in [0..Umax].
    Returns a list of tuples of user indices.
    """
    subsets = [tuple()]  # allow empty subset (no user on this RB)
    for r in range(1, Umax + 1):
        subsets += list(itertools.combinations(range(K), r))
    return subsets


def precompute_rb_costs(Hn_list, subsets, gamma_lin, sigma2,
                        P_rb_max=None, solver="MOSEK"):
    """
    For each RB and each subset, precompute the minimal power to meet SINR.
    Returns: costs[n][s_idx] = power (float), and feasibility flags.
    """
    NRB = len(Hn_list)
    costs = []
    feasible = []
    for n in range(NRB):
        H = Hn_list[n]   # [K, Nt]
        cn = []
        fn = []
        for S in subsets:
            if len(S) == 0:
                # Empty subset -> 0 power, trivially feasible
                cn.append(0.0)
                fn.append(True)
                continue

            H_sub = H[list(S), :]
            gam_sub = gamma_lin[list(S)]
            Pn, _ = solve_min_power_sinr(H_sub, gam_sub, sigma2,
                                         P_rb_max, solver)
            if np.isfinite(Pn):
                cn.append(Pn)
                fn.append(True)
            else:
                # Infeasible -> large penalty
                cn.append(1e9)
                fn.append(False)

        costs.append(cn)
        feasible.append(fn)
    return costs, feasible


def solve_scheduling_ILP(costs, subsets, K, NRB, Ptot=None):
    """
    ILP:
      variables x[n,s] in {0,1} choose exactly ONE subset per RB
      each user appears in at MOST one chosen subset overall
      objective = sum_n sum_s cost[n,s] * x[n,s]
      optional: sum costs <= Ptot
    """
    prob = pulp.LpProblem("P2_Scheduling", pulp.LpMinimize)
    x = {}
    for n in range(NRB):
        for s_idx, S in enumerate(subsets):
            x[(n, s_idx)] = pulp.LpVariable(
                f"x_{n}_{s_idx}", lowBound=0, upBound=1, cat="Binary"
            )

    # Objective
    prob += pulp.lpSum(
        costs[n][s_idx] * x[(n, s_idx)]
        for n in range(NRB)
        for s_idx, _ in enumerate(subsets)
    )

    # One subset per RB
    for n in range(NRB):
        prob += pulp.lpSum(
            x[(n, s_idx)] for s_idx, _ in enumerate(subsets)
        ) == 1

    # Each user at most once overall  ❌
    # for u in range(K):
    #     prob += pulp.lpSum(x[(n, s_idx)] 
    #                        for n in range(NRB) 
    #                        for s_idx,S in enumerate(subsets) if u in S) <= 1

    # ✅ Each user MUST be scheduled exactly once (Rmax = 1)
    for u in range(K):
        prob += pulp.lpSum(
            x[(n, s_idx)]
            for n in range(NRB)
            for s_idx, S in enumerate(subsets) if u in S
        ) == 1

    # Optional global power budget
    if Ptot is not None:
        prob += pulp.lpSum(
            costs[n][s_idx] * x[(n, s_idx)]
            for n in range(NRB)
            for s_idx, _ in enumerate(subsets)
        ) <= Ptot

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
        return None, np.inf

    chosen = []
    total_cost = 0.0
    for n in range(NRB):
        for s_idx, _ in enumerate(subsets):
            if pulp.value(x[(n, s_idx)]) > 0.5:
                chosen.append(subsets[s_idx])
                total_cost += costs[n][s_idx]
                break
    return chosen, float(total_cost)


def main():
    # ----------- System params -----------
    Nt = 4
    K = 6
    NRB = 3
    Umax = 3
    BRB = 180e3            # Hz
    N0 = 1e-17             # W/Hz (example)
    sigma2 = N0 * BRB      # noise per RB
    Ptot = 5.0             # total BS budget [W], set None to ignore
    P_rb_max = None        # per-RB cap, e.g., Ptot/NRB

    rng = np.random.default_rng(2025)

    # QoS thresholds per user (bps)
    R_th = np.full(K, 1.0e6)   # e.g., 1 Mbps per user
    gamma_lin = 2.0**(R_th / BRB) - 1.0

    # Large-scale gains (optionally distance-based); here = 1
    betas = np.ones(K)

    # ----------- Channels -----------
    Hn_list = effective_channels_per_RB(K, Nt, NRB, betas, rng=rng)  # list of [K,Nt]

    # ----------- Enumerate candidate subsets per RB -----------
    subsets = all_subsets_users(K, Umax)  # tuples of user indices

    # ----------- Precompute RB costs (convex P2 for each subset) -----------
    costs, feasible = precompute_rb_costs(Hn_list, subsets, gamma_lin, sigma2, P_rb_max)

    # ----------- Solve global ILP for scheduling -----------
    chosen_subsets, total_power = solve_scheduling_ILP(costs, subsets, K, NRB, Ptot)

    if chosen_subsets is None:
        print("No feasible schedule found (ILP infeasible).")
        return

    print("Chosen subsets per RB:", chosen_subsets)
    print("Total transmit power (W):", total_power)

    # ----------- Build final beamformers and verify sum-power <= Ptot -----------
    schedules = [list(S) for S in chosen_subsets]
    Psum, Wn_list, ok = solve_p2_given_schedule(
        Hn_list, schedules, gamma_lin, sigma2, P_rb_max
    )
    print("Verify convex solve on chosen schedule -> power:", Psum, "feasible:", ok)

    # ----------- Compute per-RB powers -----------
    rb_powers = [np.sum(np.abs(Wn) ** 2) for Wn in Wn_list]

    # ----------- Save figure -----------
    os.makedirs("figs", exist_ok=True)
    plt.figure()
    plt.bar(range(NRB), rb_powers)
    plt.xlabel("RB index")
    plt.ylabel("Power per RB (W)")
    plt.title("P2 — Power per RB for chosen schedule")
    plt.savefig("figs/p2_power_per_rb.png", dpi=300)
    plt.close()

    # ----------- Save CSV -----------
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "p2_rb_powers.csv")
    with open(csv_path, "w") as f:
        f.write("RB_index,power_W\n")
        for n, p in enumerate(rb_powers):
            f.write(f"{n},{p}\n")

    print(f"Saved figure to figs/p2_power_per_rb.png")
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
