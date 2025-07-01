import numpy as np
from quantecon.optimize import brentq
from numba import njit
from Bequests_ss import OLGModel, c_root, get_prices, get_aggregates, get_BQ, solve_SS

@njit
def solve_HH_transition(OLG, r_vec, w_vec, BQ_vec):
    # solve household problem on transition path using endogenous grid method
    b_grid = OLG.b_grid
    BQ_received = BQ_vec * OLG.zeta / OLG.omega

    # initialize policy functions on grid
    b_policy = np.zeros((OLG.S, OLG.nb, OLG.nz))
    c_policy = np.zeros((OLG.S, OLG.nb, OLG.nz))
    n_policy = np.zeros((OLG.S, OLG.nb, OLG.nz))

    # start at the end of life                      
    for z_index, z in enumerate(OLG.z_grid):
        coef = w_vec[-1] * z * OLG.eta[-1] * (1 - OLG.theta)
        for b_index, b in enumerate(b_grid):
            # use root finder to solve labor-consumption problem at end of life
            args = (OLG, r_vec[-1], coef, b, BQ_received[-1])
            c = brentq(c_root, 0, (1+r_vec[-1])*b + coef + BQ_received[-1], args).root
            n = OLG.n_from_c(c, coef)
            c_policy[-1, b_index, z_index] = c
            n_policy[-1, b_index, z_index] = n
            b_policy[-1, b_index, z_index] = (1+r_vec[-1]) * b + coef * n + BQ_received[-1] - c
    
    # iterate backwards with Euler equation
    for s in range(OLG.S-2, -1, -1):
        r = r_vec[s]
        r_next = r_vec[s+1]
        w = w_vec[s]
        for z_index, z in enumerate(OLG.z_grid):
            coef = w * z * OLG.eta[s] * (1 - OLG.theta)
            # find the expected marginal utility of consumption in the next period
            E_mu_c = np.zeros(OLG.nb)
            for zp_index in range(OLG.nz):
                E_mu_c += OLG.pi[z_index, zp_index] * OLG.mu_c(c_policy[s+1, :, zp_index])
            c = OLG.inv_mu_c(OLG.rho[s]*OLG.mu_b(b_grid) + (1-OLG.rho[s])*OLG.beta*(1 + r_next)*E_mu_c)
            n = OLG.n_from_c(c, coef)
            new_grid = (b_grid + c - coef * n - BQ_received[s]) / (1 + r)
            c_policy[s, :, z_index] = np.interp(b_grid, new_grid, c)
            n_policy[s, :, z_index] = OLG.n_from_c(c_policy[s, :, z_index], coef)
            b_policy[s, :, z_index] = (1+r) * b_grid + coef*n_policy[s, :, z_index] + BQ_received[s] - c_policy[s, :, z_index]

    return b_policy, c_policy, n_policy

@njit
def iterate_distribution(OLG, h_init, b_policies):
    """
    Iterate the distribution of agents forward by one period.
    """
    h_next = np.zeros((OLG.S, OLG.nb, OLG.nz))
    
    # New born agents
    h_next[0, 0, :] = OLG.initial_dist * OLG.omega[0]

    # Update distribution for existing agents
    for s in range(OLG.S - 1):
        for b_index in range(OLG.nb):
            for z_index in range(OLG.nz):
                for z_next in range(OLG.nz):
                    b_next = b_policies[s, b_index, z_index]
                    
                    if b_next <= OLG.b_min:
                        h_next[s+1, 0, z_next] += h_init[s, b_index, z_index] * OLG.pi[z_index, z_next] * (1 - OLG.rho[s]) / (1 + OLG.g_n)
                    elif b_next >= OLG.b_max:
                        h_next[s+1, -1, z_next] += h_init[s, b_index, z_index] * OLG.pi[z_index, z_next] * (1 - OLG.rho[s]) / (1 + OLG.g_n)
                    else:
                        idx = np.searchsorted(OLG.b_grid, b_next)
                        weight_high = (b_next - OLG.b_grid[idx-1]) / (OLG.b_grid[idx] - OLG.b_grid[idx-1])
                        weight_low = 1 - weight_high
                        h_next[s+1, idx-1, z_next] += weight_low * h_init[s, b_index, z_index] * OLG.pi[z_index, z_next] * (1 - OLG.rho[s]) / (1 + OLG.g_n)
                        h_next[s+1, idx, z_next] += weight_high * h_init[s, b_index, z_index] * OLG.pi[z_index, z_next] * (1 - OLG.rho[s]) / (1 + OLG.g_n)
    
    return h_next

@njit
def get_cross_sectional_policies(OLG, policy_matrix, t):
    '''
    Get the policy functions for each living cohort at time t on the transition path.
    '''
    policies = np.zeros((OLG.S, OLG.nb, OLG.nz))

    for s in range(OLG.S):
        policies[s] = policy_matrix[t + OLG.S -1 - s, s]
    
    return policies

@njit
def solve_transition_path(OLG, T, initial_state, tol=1e-5, maxiter=500, xi=0.25):

    K_0, L_0, BQ_0, h_0 = initial_state

    # Solve for steady state
    K_ss, L_ss, BQ_ss, r_ss, w_ss = solve_SS(OLG, tol=tol, max_iter=maxiter, xi=xi, initial_guess=(K_0, L_0, BQ_0))

    # Get initial prices
    r_0, w_0 = get_prices(OLG, K_0, L_0)

    # Initialize guess of the transition path
    K_path = np.linspace(K_0, K_ss, T)
    L_path = np.linspace(L_0, L_ss, T)
    BQ_path = np.linspace(BQ_0, BQ_ss, T)
    r_path = np.linspace(r_0, r_ss, T)
    w_path = np.linspace(w_0, w_ss, T)

    # we have cohorts who start lives before the transition
    num_cohorts = T + OLG.S - 1

    error = tol * 100
    m = 0

    while error > tol:
        m += 1
        
        # Extend price paths for incomplete lifetimes on transition path
        r_extended = np.concatenate((np.repeat(r_0, OLG.S-1), r_path, np.repeat(r_ss, OLG.S-1)))
        w_extended = np.concatenate((np.repeat(w_0, OLG.S-1), w_path, np.repeat(w_ss, OLG.S-1)))
        BQ_extended = np.concatenate((np.repeat(BQ_0, OLG.S-1), BQ_path, np.repeat(BQ_ss, OLG.S-1)))

        # matrices to store policy functions for all cohorts alive on transition path
        b_mat = np.zeros((num_cohorts, OLG.S, OLG.nb, OLG.nz))
        c_mat = np.zeros((num_cohorts, OLG.S, OLG.nb, OLG.nz))
        n_mat = np.zeros((num_cohorts, OLG.S, OLG.nb, OLG.nz))

        for j in range(num_cohorts):
            r_vec = r_extended[j:(j+OLG.S)]
            w_vec = w_extended[j:(j+OLG.S)]
            BQ_vec = BQ_extended[j:(j+OLG.S)]
            b_mat[j], c_mat[j], n_mat[j] = solve_HH_transition(OLG, r_vec, w_vec, BQ_vec)

        K_path_next = np.zeros(T)
        L_path_next = np.zeros(T)
        BQ_path_next = np.zeros(T)
        K_path_next[0] = K_0
        L_path_next[0] = L_0
        BQ_path_next[0] = BQ_0

        # Iterate distribution using policy functions
        h = h_0
        for t in range(T-1):
            b_policies = get_cross_sectional_policies(OLG, b_mat, t)
            n_policies = get_cross_sectional_policies(OLG, n_mat, t+1)

            # Iterate distribution forward
            h_next = iterate_distribution(OLG, h, b_policies)

            # Compute aggregates
            K_path_next[t+1], L_path_next[t+1] = get_aggregates(OLG, h_next, n_policies)
            BQ_path_next[t+1] = get_BQ(OLG, b_policies, h, r_path[t+1])
            h = h_next

        # Compute error
        error = max(np.max(np.abs(K_path_next - K_path)), np.max(np.abs(L_path_next - L_path)), np.max(np.abs(BQ_path_next - BQ_path)))

        # Update paths
        K_path = (1-xi)*K_path + xi*K_path_next
        L_path = (1-xi)*L_path + xi*L_path_next
        BQ_path = (1-xi)*BQ_path + xi*BQ_path_next
        for t in range(T):
            r_path[t], w_path[t] = get_prices(OLG, K_path[t], L_path[t])
            
        if m > maxiter:
            print('No convergence')
            print('Current path:')
            print(K_path_next, L_path_next, BQ_path_next)
            return
    
    print(f'Transition path converged in {m} iterations')
    return K_path, L_path, BQ_path, r_path, w_path