# sim/p2_solver.py
import numpy as np
import cvxpy as cp


def solve_min_power_sinr(H_sub, gamma_sub, sigma2, P_rb_max=None, solver=None):
    """
    Minimize total transmit power for a single RB and a given subset of users,
    subject to SINR >= gamma_sub (per user), using an SOCP formulation.

    H_sub : [U, Nt] complex numpy array (effective channels of scheduled users)
    gamma_sub : [U] array of target SINR (linear scale)
    sigma2 : noise power on this RB
    P_rb_max : optional per-RB power cap (float) or None
    solver : optional string, ignored unless a valid cvxpy solver is provided.
             If solver == "MOSEK" or None or invalid, we fallback to SCS.

    Returns:
        P_opt (float): minimal total power (sum |W|^2), or np.inf if infeasible
        W_opt (np.ndarray): [Nt, U] complex optimal beamforming matrix
                            (zeros if infeasible)
    """
    H_sub = np.asarray(H_sub)
    gamma_sub = np.asarray(gamma_sub)

    U, Nt = H_sub.shape
    # Beamforming matrix: columns = w_u for each user u in this subset
    W = cp.Variable((Nt, U), complex=True)

    # hW[u, v] = h_u^H w_v  (size [U, U])
    hW = H_sub @ W

    constraints = []

    # Per-user SINR constraints via SOCP
    for u in range(U):
        gamma_u = float(gamma_sub[u])
        if gamma_u <= 0:
            # gamma <= 0 means no SINR requirement; we just skip constraint
            continue

        # Desired signal: h_u^H w_u
        sig_u = hW[u, u]

        # Interference terms: h_u^H w_v for v != u
        interf_terms = []
        for v in range(U):
            if v == u:
                continue
            interf_terms.append(hW[u, v])

        # Add noise term sqrt(sigma2) as last element in the norm
        noise_scalar = np.sqrt(sigma2)

        # Build SOCP constraint:
        # || [interferences, sqrt(sigma2)] ||_2 <= (1/sqrt(gamma_u)) * Re(sig_u)
        soc_vec = cp.hstack(interf_terms + [noise_scalar])

        # Phase fixing & non-negative real part to make SINR constraint convex
        constraints += [
            cp.imag(sig_u) == 0,                  # align phase
            cp.real(sig_u) >= 0,                  # non-negative
            cp.norm(soc_vec, 2) <=
            (1.0 / np.sqrt(gamma_u)) * cp.real(sig_u)
        ]

    # Total power = sum |W|^2  (Frobenius norm squared)
    total_power = cp.sum_squares(cp.abs(W))

    # Optional per-RB power cap
    if P_rb_max is not None:
        constraints.append(total_power <= P_rb_max)

    # Define problem
    prob = cp.Problem(cp.Minimize(total_power), constraints)

    # Choose solver: if MOSEK not installed, fallback to SCS
    solver_used = None
    try:
        if solver is None or solver == "MOSEK":
            solver_used = cp.SCS
        else:
            solver_used = solver
        prob.solve(solver=solver_used, verbose=False)
    except Exception:
        # If something goes wrong, fallback once more to SCS
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            return np.inf, np.zeros((Nt, U), dtype=np.complex128)

    if prob.status not in ["optimal", "optimal_inexact"]:
        # Infeasible or failed
        return np.inf, np.zeros((Nt, U), dtype=np.complex128)

    # Extract optimal beams
    W_opt = W.value
    if W_opt is None:
        return np.inf, np.zeros((Nt, U), dtype=np.complex128)

    P_opt = np.sum(np.abs(W_opt) ** 2).real
    return float(P_opt), W_opt


def solve_p2_given_schedule(Hn_list, schedules, gamma_lin, sigma2,
                            P_rb_max=None, solver=None):
    """
    Solve P2 for a GIVEN global schedule (which users on which RB).

    Hn_list : list of length NRB, each element [K, Nt] complex (all users' channels on RB n)
    schedules : list of length NRB, each element is a list of user indices scheduled on RB n
    gamma_lin : [K] array of SINR targets (linear)
    sigma2 : noise power on each RB
    P_rb_max : optional per-RB power cap
    solver : forwarded to solve_min_power_sinr (see there)

    Returns:
        P_sum (float): total power over all RBs (sum_n sum |W_n|^2)
        Wn_list (list): list of beamforming matrices per RB.
                        Wn_list[n] is [Nt, U_n] complex for RB n.
        feasible (bool): True if all RBs are feasible, False otherwise.
    """
    NRB = len(Hn_list)
    assert len(schedules) == NRB, "schedules must have length NRB"

    Wn_list = []
    P_sum = 0.0
    feasible = True

    for n in range(NRB):
        H_all = Hn_list[n]           # [K, Nt]
        users_n = schedules[n]       # list of user indices for RB n

        if len(users_n) == 0:
            # No users scheduled on this RB -> zero beams
            Nt = H_all.shape[1]
            Wn = np.zeros((Nt, 0), dtype=np.complex128)
            Wn_list.append(Wn)
            continue

        H_sub = H_all[users_n, :]              # [U_n, Nt]
        gamma_sub = gamma_lin[users_n]         # [U_n]

        Pn, Wn = solve_min_power_sinr(H_sub, gamma_sub, sigma2,
                                      P_rb_max=P_rb_max,
                                      solver=solver)

        if not np.isfinite(Pn):
            feasible = False
        else:
            P_sum += Pn

        Wn_list.append(Wn)

    return P_sum, Wn_list, feasible
