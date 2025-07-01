Certainly! Here's the revised version with math wrapped in "$" for markdown compatibility:

# Overlapping Generations Model with Endogenous Labor Supply

This model describes an economy with S-period lived agents and endogenous labor supply. It's based on the framework presented in Chapter 4 of the OG textbook.

## Households

A unit measure of identical individuals are born each period and live for S periods. Households choose lifetime consumption $\{c_{s,t+s-1}\}_{s=1}^S$, labor supply $\{n_{s,t+s-1}\}_{s=1}^S$, and savings $\{b_{s+1,t+s}\}_{s=1}^{S-1}$ to maximize lifetime utility:

$\max_{\{c_{s,t+s-1}, n_{s,t+s-1}\}_{s=1}^S, \{b_{s+1,t+s}\}_{s=1}^{S-1}} \sum_{s=1}^S \beta^{s-1} u(c_{s,t+s-1}, n_{s,t+s-1})$

subject to the budget constraints:

$c_{s,t} + b_{s+1,t+1} = (1 + r_t)b_{s,t} + w_t n_{s,t}$

with $b_{1,t} = b_{S+1,t} = 0$

The period utility function uses an elliptical disutility of labor:

$u(c_{s,t}, n_{s,t}) = \frac{c_{s,t}^{1-\sigma} - 1}{1 - \sigma} + \chi_s^n b[1 - (\frac{n_{s,t}}{\tilde{l}})^\upsilon]^{\frac{1}{\upsilon}}$

where $\tilde{l}$ is the time endowment, $b$ and $\upsilon$ are parameters of the elliptical disutility function, and $\chi_s^n$ is an age-specific disutility of labor parameter.

## Firms

Firms use a Cobb-Douglas production function:

$Y_t = A K_t^\alpha L_t^{1-\alpha}$

Profit maximization leads to the following factor prices:

$r_t = \alpha A (\frac{L_t}{K_t})^{1-\alpha} - \delta$
$w_t = (1-\alpha) A (\frac{K_t}{L_t})^\alpha$

where $\delta$ is the depreciation rate.

## Market Clearing

The labor, capital, and goods markets must clear:

$L_t = \sum_{s=1}^S n_{s,t}$
$K_t = \sum_{s=2}^S b_{s,t}$
$Y_t = C_t + I_t$

where $I_t \equiv K_{t+1} - (1-\delta)K_t$ is investment.

## Equilibrium

A steady-state equilibrium is defined as constant allocations of consumption $\{\bar{c}_s\}_{s=1}^S$, labor supply $\{\bar{n}_s\}_{s=1}^S$, and savings $\{\bar{b}_s\}_{s=2}^S$, and prices $\bar{r}$ and $\bar{w}$ such that:

1. Households optimize according to their Euler equations.
2. Firms optimize according to their first-order conditions.
3. Markets clear.

A non-steady-state equilibrium is defined as time paths of allocations and prices that satisfy the same conditions, with households having rational expectations about the evolution of the distribution of savings.

## Solution Method

The model is solved using:

1. For the steady state: A nested fixed point algorithm with an outer loop for the interest rate and an inner loop for household decisions.
2. For the transition path: Time path iteration (TPI) to find the equilibrium path from an initial state to the steady state.

Both methods use numerical optimization techniques to solve the household problem and iterate until convergence is achieved.


---
## Transition Path pseudocode:


```
Function SolveTransitionPath(initial_state, steady_state, T, parameters):
    # initial_state: initial distribution of capital
    # steady_state: steady-state values from SolveSteadyState
    # T: number of periods to simulate

    Initialize guess for time path of interest rates {r_t} from t=0 to T
    Set final values of {r_t} to steady_state.r

    While not converged:
        Calculate implied time path of wages {w_t} from {r_t}
        
        # Solve household problem for each cohort
        For each cohort alive at t=0 or born during transition:
            Initialize guesses for lifetime {n_s,t} and {b_s,t}
            Define CohortEulerSystem({n_s,t}, {b_s,t}, {r_t}, {w_t}):
                For each period in cohort's lifetime:
                    Calculate c_s,t using budget constraint
                    Calculate labor supply Euler error
                    If not last period of life:
                        Calculate savings Euler error
                Return all Euler errors
            
            Use root-finding algorithm to solve CohortEulerSystem

        # Aggregate individual decisions
        For t = 0 to T:
            K_t = sum(b_s,t for all s alive at t)
            L_t = sum(n_s,t for all s alive at t)

        # Check market clearing and update prices
        For t = 0 to T:
            r_new_t = MPK(K_t, L_t) - delta
        
        If max(|r_new_t - r_t|) < tolerance for all t:
            Converged = True
        Else:
            Update {r_t} = xi * {r_new_t} + (1-xi) * {r_t}
    
    # Calculate other time paths
    For t = 0 to T:
        Y_t = F(K_t, L_t)
        C_t = Y_t - (K_t+1 - (1-delta)*K_t)
    
    Return {r_t}, {w_t}, {K_t}, {L_t}, {Y_t}, {C_t}, individual decisions

# Main program
parameters = SetModelParameters()
steady_state = SolveSteadyState(parameters)
initial_state = SetInitialState()
T = ChooseNumberOfPeriods()
transition_path = SolveTransitionPath(initial_state, steady_state, T, parameters)
```


# Conesa & Krueger TPI pseudocode:

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



# Conesa & Kruger TPI (Dean):

Algorithm to compute for the transition path given steady state values $(K_{ss}, L_{ss}, h_{ss})$ and an initial state $(K_0, L_0, h_0)$:

1. Choose T – the number of periods to reach the final steady state.
2. Make an initial guess for the sequence $\{K_t, L_t\}$. We guess a linear interpolation between $\{K_{0}, L_{0}\}$ and $\{K_{ss}, L_{ss}\}$.




# Cross sectional policies

This function should take a matrix of policy functions for all cohorts and return a list of policy functions for each living cohort at time t.



# EGM with bequests

# Flow utility:

$u(c_t, n_t, b_{t+1}) = \frac{1-\sigma}{1-\sigma} - \chi^{-\sigma} \frac{n_t^{1+\psi}}{1+\psi} + \chi^b b_{t+1}^{1-\sigma} \frac{1}{1-\sigma}$

$\rightarrow$ using FOC, can write as function of just 2 variables, $b_{t+1}$, $n_t$:

$u(n_t, b_{t+1}) = \frac{(\omega_t e_t n_t + R_t b_t - b_{t+1})^{1-\sigma}}{1-\sigma} - \chi^{-\sigma} \frac{n_t^{1+\psi}}{1+\psi} + \chi^b b_{t+1}^{1-\sigma} \frac{1}{1-\sigma}$

$\frac{\partial u}{\partial n_t} = \omega_t e_t (\omega_t e_t n_t + R_t b_t)^{-\sigma} - \chi^{-\sigma} n_t^\psi = 0$

$\rightarrow \omega_t e_t c_t^{-\sigma} = \chi^{-\sigma} n_t^\psi$

$\rightarrow n_t(c_t) = \left(\frac{\omega_t e_t c_t^{-\sigma}}{\chi^{-\sigma}}\right)^{\frac{1}{\psi}}$
