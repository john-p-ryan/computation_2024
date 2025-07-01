from EGM_ss import (
    OLGModel,
    constrained_c_root,
    get_prices,
    get_aggregates,
    solve_SS,
)
from quantecon.optimize import brentq
import numpy as np
from numba import njit


@njit
def solve_HH_transition(OLG, r_vec, w_vec, b_vec):
    # solve household problem on transition path using endogenous grid method
    a_grid = OLG.a_grid

    # initialize policy functions on grid
    a_policy_w = np.zeros(
        (OLG.J_r - 1, OLG.nz, OLG.na)
    )  # savings policy function for workers
    c_policy_w = np.zeros(
        (OLG.J_r - 1, OLG.nz, OLG.na)
    )  # consumption policy function for workers
    l_policy = np.zeros(
        (OLG.J_r - 1, OLG.nz, OLG.na)
    )  # labor policy function for workers
    a_policy_r = np.zeros(
        (OLG.N - OLG.J_r + 1, OLG.na)
    )  # savings policy function for retirees
    c_policy_r = np.zeros(
        (OLG.N - OLG.J_r + 1, OLG.na)
    )  # consumption policy function for retirees

    # start at the end of life
    c_policy_r[-1, :] = a_grid * (1 + r_vec[-1]) + b_vec[-1]

    # iterate backward with Euler equation for retirees
    for j in range(OLG.N - OLG.J_r - 1, -1, -1):
        t = OLG.N - (OLG.N - OLG.J_r - j)
        r = r_vec[t]
        r_next = r_vec[t + 1]
        b = b_vec[t]
        c_r = OLG.inv_mu_c(
            OLG.beta * (1 + r_next) * OLG.mu_c(c_policy_r[j + 1, :])
        )
        # find asset holdings from BC
        new_grid = (a_grid + c_r - b) / (1 + r)
        c_policy_r[j, :] = np.interp(a_grid, new_grid, c_r)
        a_policy_r[j, :] = (1 + r) * a_grid + b - c_policy_r[j, :]

    # start just before retirement
    # find c with Euler equation, find l with FOC
    t = OLG.N - OLG.J_r
    r = r_vec[t]
    w = w_vec[t]
    for z_index, z in enumerate(OLG.z_grid):
        coef = w * z * OLG.eta[-1] * (1 - OLG.theta)
        c_w = OLG.inv_mu_c(
            OLG.beta * (1 + r_vec[t + 1]) * OLG.mu_c(c_policy_r[0, :])
        )
        l_w = OLG.l_from_c(c_w, coef)
        # find asset holdings from BC
        new_grid = (a_grid + c_w - coef * l_w) / (1 + r)
        c_policy_w[-1, z_index, :] = np.interp(a_grid, new_grid, c_w)
        l_policy[-1, z_index, :] = OLG.l_from_c(
            c_policy_w[-1, z_index, :], coef
        )
        a_policy_w[-1, z_index, :] = (
            (1 + r) * a_grid
            + coef * l_policy[-1, z_index, :]
            - c_policy_w[-1, z_index, :]
        )

    # iterate backwards with Euler equation for workers
    for j in range(OLG.J_r - 3, -1, -1):
        t = OLG.N - OLG.J_r + j
        r = r_vec[t]
        w = w_vec[t]
        for z_index, z in enumerate(OLG.z_grid):
            coef = w * z * OLG.eta[j] * (1 - OLG.theta)
            # find the expected marginal utility of consumption in the next period
            E_mu_c = np.zeros(OLG.na)
            for zp_index in range(OLG.nz):
                E_mu_c += OLG.pi[z_index, zp_index] * OLG.mu_c(
                    c_policy_w[j + 1, zp_index, :]
                )
            c_w = OLG.inv_mu_c(OLG.beta * (1 + r_vec[t + 1]) * E_mu_c)
            l_w = OLG.l_from_c(c_w, coef)
            new_grid = (a_grid + c_w - coef * l_w) / (1 + r)
            c_policy_w[j, z_index, :] = np.interp(a_grid, new_grid, c_w)
            l_policy[j, z_index, :] = OLG.l_from_c(
                c_policy_w[j, z_index, :], coef
            )
            a_policy_w[j, z_index, :] = (
                (1 + r) * a_grid
                + coef * l_policy[j, z_index, :]
                - c_policy_w[j, z_index, :]
            )
            # replace policies less than a_min with a_min for borrowing constraint
            # find the first index where borrowing constraint is not binding
            last_binding = np.argmax(a_policy_w[j, z_index, :] >= OLG.a_min)
            a_policy_w[j, z_index, :last_binding] = OLG.a_min
            # loop over indices where binding
            for i in range(last_binding):
                if l_policy[j, z_index, i] == 1:
                    # if binding and l = 1, then just use BC
                    c_policy_w[j, z_index, i] = (
                        (1 + r) * a_grid[i] + coef - OLG.a_min
                    )
                else:
                    # essentially perform grid search here
                    args = (OLG, r, a_grid[i], coef)
                    # use root finder to find c that satisfies BC
                    c_policy_w[j, z_index, i] = brentq(
                        constrained_c_root, 0, c_policy_w[j, z_index, i], args
                    ).root
                    l_policy[j, z_index, i] = OLG.l_from_c(
                        c_policy_w[j, z_index, i], coef
                    )

    return a_policy_w, a_policy_r, c_policy_w, c_policy_r, l_policy


def iterate_distribution(OLG, h_w_init, h_r_init, a_w_policies, a_r_policies):
    """
    Iterate the distribution of agents forward by one period.

    Args:
    OLG: OLGModel object
    h_w_init: Initial distribution of workers (J_r-1, na, nz)
    h_r_init: Initial distribution of retirees (N-J_r+1, na)
    a_w_policies: Savings policy function for workers (J_r-1, nz, na)
    a_r_policies: Savings policy function for retirees (N-J_r+1, na)

    Returns:
    h_w_next: Distribution of workers in the next period (J_r-1, na, nz)
    h_r_next: Distribution of retirees in the next period (N-J_r+1, na)
    """
    h_w_next = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))
    h_r_next = np.zeros((OLG.N - OLG.J_r + 1, OLG.na))

    # New born workers
    h_w_next[0, 0, :] = OLG.initial_dist * OLG.mu[0]

    # Update distribution for existing workers
    for j in range(1, OLG.J_r - 1):
        for a_k in range(OLG.na):
            for z in range(OLG.nz):
                for z_next in range(OLG.nz):
                    a_next = a_w_policies[j - 1, z, a_k]

                    if a_next <= OLG.a_min:
                        h_w_next[j, 0, z_next] += (
                            h_w_init[j - 1, a_k, z]
                            * OLG.pi[z, z_next]
                            / (1 + OLG.n)
                        )
                    elif a_next >= OLG.a_max:
                        h_w_next[j, -1, z_next] += (
                            h_w_init[j - 1, a_k, z]
                            * OLG.pi[z, z_next]
                            / (1 + OLG.n)
                        )
                    else:
                        idx = np.searchsorted(OLG.a_grid, a_next)
                        weight_high = (a_next - OLG.a_grid[idx - 1]) / (
                            OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                        )
                        weight_low = 1 - weight_high
                        h_w_next[j, idx - 1, z_next] += (
                            weight_low
                            * h_w_init[j - 1, a_k, z]
                            * OLG.pi[z, z_next]
                            / (1 + OLG.n)
                        )
                        h_w_next[j, idx, z_next] += (
                            weight_high
                            * h_w_init[j - 1, a_k, z]
                            * OLG.pi[z, z_next]
                            / (1 + OLG.n)
                        )

    # Transition from workers to retirees
    for a_k in range(OLG.na):
        for z in range(OLG.nz):
            a_next = a_w_policies[-1, z, a_k]

            if a_next <= OLG.a_min:
                h_r_next[0, 0] += h_w_init[-1, a_k, z] / (1 + OLG.n)
            elif a_next >= OLG.a_max:
                h_r_next[0, -1] += h_w_init[-1, a_k, z] / (1 + OLG.n)
            else:
                idx = np.searchsorted(OLG.a_grid, a_next)
                weight_high = (a_next - OLG.a_grid[idx - 1]) / (
                    OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                )
                weight_low = 1 - weight_high
                h_r_next[0, idx - 1] += (
                    weight_low * h_w_init[-1, a_k, z] / (1 + OLG.n)
                )
                h_r_next[0, idx] += (
                    weight_high * h_w_init[-1, a_k, z] / (1 + OLG.n)
                )

    # Update distribution for existing retirees
    for j in range(1, OLG.N - OLG.J_r + 1):
        for a_k in range(OLG.na):
            a_next = a_r_policies[j - 1, a_k]

            if a_next <= OLG.a_min:
                h_r_next[j, 0] += h_r_init[j - 1, a_k] / (1 + OLG.n)
            elif a_next >= OLG.a_max:
                h_r_next[j, -1] += h_r_init[j - 1, a_k] / (1 + OLG.n)
            else:
                idx = np.searchsorted(OLG.a_grid, a_next)
                weight_high = (a_next - OLG.a_grid[idx - 1]) / (
                    OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                )
                weight_low = 1 - weight_high
                h_r_next[j, idx - 1] += (
                    weight_low * h_r_init[j - 1, a_k] / (1 + OLG.n)
                )
                h_r_next[j, idx] += (
                    weight_high * h_r_init[j - 1, a_k] / (1 + OLG.n)
                )

    # Normalize the distributions
    total_mass = np.sum(h_w_next) + np.sum(h_r_next)
    h_w_next /= total_mass
    h_r_next /= total_mass

    return h_w_next, h_r_next


@njit
def get_cross_section(OLG, a_w_mat, a_r_mat, l_mat, t):
    # get cross section of policy functions at a point in the transition path

    # Initialize arrays to store the cross-section policies
    a_policies_w = np.zeros((OLG.J_r - 1, OLG.nz, OLG.na))
    a_policies_r = np.zeros((OLG.N - OLG.J_r + 1, OLG.na))
    l_policies = np.zeros((OLG.J_r - 1, OLG.nz, OLG.na))

    # Extract policies for workers
    for j in range(OLG.J_r - 1):
        cohort_t = t - j  # The time period when this cohort was born
        if 0 <= cohort_t < len(a_w_mat):
            a_policies_w[j] = a_w_mat[cohort_t, j]
            l_policies[j] = l_mat[cohort_t, j]

    # Extract policies for retirees
    for j in range(OLG.N - OLG.J_r + 1):
        cohort_t = t - (
            OLG.J_r - 1 + j
        )  # The time period when this cohort was born
        if 0 <= cohort_t < len(a_r_mat):
            a_policies_r[j] = a_r_mat[cohort_t, j]

    return a_policies_w, a_policies_r, l_policies


@njit
def SolveTransitionPath(
    OLG, T, K_0, L_0, h_w_0, h_r_0, tol=1e-5, maxiter=200, rho=0.1
):
    # Solve for steady state
    K_ss, L_ss, r_ss, w_ss, b_ss = solve_SS(
        OLG, tol=tol, maxiter=maxiter, rho=rho, K0=K_0, L0=L_0
    )

    # Get initial prices
    r_0, w_0, b_0 = get_prices(OLG, K_0, L_0)

    # Initialize guess of the transition path
    K_path = np.linspace(K_0, K_ss, T)
    L_path = np.linspace(L_0, L_ss, T)
    r_path = np.linspace(r_0, r_ss, T)
    w_path = np.linspace(w_0, w_ss, T)
    b_path = np.linspace(b_0, b_ss, T)

    T_extended = T + 2 * OLG.N
    error = tol * 100
    n = 0

    while error > tol:
        n += 1

        # Extend price paths
        r_extended = np.concatenate(
            [np.repeat(r_0, OLG.N), r_path, np.repeat(r_ss, OLG.N)]
        )
        w_extended = np.concatenate(
            [np.repeat(w_0, OLG.N), w_path, np.repeat(w_ss, OLG.N)]
        )
        b_extended = np.concatenate(
            [np.repeat(b_0, OLG.N), b_path, np.repeat(b_ss, OLG.N)]
        )

        a_w_mat = np.zeros((T_extended, OLG.J_r - 1, OLG.nz, OLG.na))
        a_r_mat = np.zeros((T_extended, OLG.N - OLG.J_r + 1, OLG.na))
        l_mat = np.zeros((T_extended, OLG.J_r - 1, OLG.nz, OLG.na))

        # Solve for HH policy functions for each cohort alive on the transition path
        for s in range(T_extended):
            r_vec = r_extended[s : (s + OLG.N)]
            w_vec = w_extended[s : (s + OLG.N)]
            b_vec = b_extended[s : (s + OLG.N)]
            a_w, a_r, _, _, l_w = solve_HH_transition(OLG, r_vec, w_vec, b_vec)
            a_w_mat[s] = a_w
            a_r_mat[s] = a_r
            l_mat[s] = l_w

        K_path_next = np.zeros(T)
        L_path_next = np.zeros(T)

        # Iterate distribution using policy functions
        h_w, h_r = h_w_0, h_r_0
        for t in range(T):
            # Get policy functions for period t
            a_policies_w = a_w_mat[t]
            a_policies_r = a_r_mat[t]
            l_policies = l_mat[t]

            # Iterate distribution forward
            h_w, h_r = iterate_distribution(
                OLG, h_w, h_r, a_policies_w, a_policies_r
            )

            # Compute aggregates
            K_path_next[t], L_path_next[t] = get_aggregates(
                OLG, h_w, h_r, l_policies
            )

        # Compute error
        error = max(
            np.max(np.abs(K_path_next - K_path)),
            np.max(np.abs(L_path_next - L_path)),
        )

        if error < tol:
            print(f"Converged in {n} iterations")
        else:
            if n > maxiter:
                print("No convergence")
                print("Current path:")
                print(K_path_next, L_path_next)
                break

            # Update paths
            K_path = (1 - rho) * K_path + rho * K_path_next
            L_path = (1 - rho) * L_path + rho * L_path_next
            for t in range(T):
                r_path[t], w_path[t], b_path[t] = get_prices(
                    OLG, K_path[t], L_path[t]
                )

    return K_path, L_path, r_path, w_path, b_path
