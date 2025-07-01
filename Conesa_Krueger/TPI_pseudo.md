
# pseudocode for Transition Path Iteration in Conesa & Krueger

```
Function SolveTransitionPath(OLG, T, K_0, L_0, h_w_0, h_r_0 tol, maxiter):
    # OLG is a class of parameters
    # T is length of transition path, N is length of lifetime
    # assume aggregates and prices are converged after T
    K_ss, L_ss, r_ss, w_ss, b_ss = SolveSS(OLG)

    r_0, w_0, b_0 = get_prices(OLG, K_0, L_0)

    # initialize a guess of the transition path
    K_path, L_path, r_path, w_path, b_path = interpolate(0, ss)

    T_extended = T + 2*N
    error = tol * 100
    n = 0
    while error < tol:
        n += 1
        r_extended = concatenate(repeat(r0, N), r_path, repeat(r_ss, N))
        w_extented = concatenate(repeat(w0, N), w_path, repeat(w_ss, N))
        b_extended = concatenate(repeat(b0, N), b_path, repeat(b_ss, N))
        a_w_mat = zeros(T_extended, J_r-1, na, nz)
        a_r_mat = zeros(T_extended, N-J_r, na)
        l_mat = zeros(T_extended, J_r-1, na, nz)
        # solve for the HH policy functions for each cohort, s, alive on the transition path
        for s in range(T_extended):
            r_vec = r_extended[s:(s+N)]
            w_vec = w_extended[s:(s+N)]
            b_vec = b_extended[s:(s+N)]
            a_w, a_r, l_w = Solve_HH_transiton(r_vec, w_vec, b_vec)
            a_w_mat[s, :, :, :] = a_w
            a_r_mat[s, :, :] = a_r
            l_mat[s, :, :] = l_w
        
        K_path_next = zeros(T)
        L_path_next = zeros(T)
        # iterate distribution using policy functions
        h_w_init, h_r_init = h_w_0, h_r_0
        for t in range(T):
            # get the policy functions in period t for all cohorts alive
            a_policies_w = cross_section(a_w_mat, t)
            a_policies_r = cross_section(a_r_mat, t)
            l_policies = cross_section(l_mat, t)
            # iterate the distribution forward
            h_w_next, h_w_next = iterate_distribution(h_w_init, h_r_init)
            K_path_next[t+1], L_path_next[t+1] = get_aggregates(h_w_next, h_r_next, l_policies)

        error = max(max(|K_path_next - K_path|), max(|L_path_next - L_path|))
        if error < tol:
            print('converged in {n} iterations')
        else:
            if n > maxiter:
                print('No convergence')
                print('Current path:')
                print(K_path_next, L_path_next)
                break
            K_path, L_path = K_path_next, L_path_next
            for t in range(T):
                r_path[t], w_path[t], b_path[t] = get_prices(K_path[t], L_path[t])
```