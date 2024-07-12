import numpy as np
from quantecon.optimize.root_finding import brentq
from numba import njit, float64, int64
from numba.experimental import jitclass

@njit
def nonuniform_grid(a, b, n=100, density=2):
    '''
    a: lower bound
    b: upper bound
    density: positive scalar for the density of the grid. > 1 for more points at the lower end
    '''
    linear_points = np.linspace(0, 1, n)
    nonlinear_points = (linear_points ** density) * (b-a) + a
    return nonlinear_points 


primitives = [
    ('beta', float64),
    ('gamma', float64),
    ('psi', float64),
    ('sigma', float64),
    ('alpha', float64),
    ('delta', float64),
    ('S', int64),
    ('g_n', float64),
    ('b_min', float64),
    ('b_max', float64),
    ('nb', int64),
    ('b_grid', float64[:]),
    ('z_grid', float64[:]),
    ('nz', int64),
    ('initial_dist', float64[:]),
    ('pi', float64[:,:]),
    ('theta', float64),
    ('eta', float64[:]),
    ('mu', float64[:])
]

@jitclass(primitives)
class OLGModel:
    def __init__(self, 
                 beta=0.97, 
                 gamma=2.50,
                 psi=2.0,
                 sigma=2.0, 
                 alpha=0.36, 
                 delta=0.06,
                 S=66, 
                 g_n=0.011,
                 b_grid=np.linspace(1e-8, 30.0, 200),
                 z_grid=np.array([2.0, 0.3]), 
                 initial_dist=np.array([0.2037, 0.7963]),
                 pi=np.array([[0.9261, 0.0739], [0.0189, 0.9811]]),
                 theta=0.11,
                 eta=np.ones(45)):
        # Primitives
        self.beta = beta                                # discount rate
        self.gamma = gamma                              # Frisch elasticity
        self.psi = psi                                  # disutility of labor
        self.sigma = sigma                              # CRRA consumption
        self.alpha = alpha                              # capital share
        self.delta = delta                              # depreciation rate
        self.S = S                                      # death age
        self.g_n = g_n                                  # population growth rate
        self.b_min = b_grid[0]                          # assets lower bound
        self.b_max = b_grid[-1]                         # assets upper bound
        self.nb = len(b_grid)                           # number of asset grid points
        self.b_grid = b_grid
        self.z_grid = z_grid                            # productivity shocks
        self.nz = len(z_grid)                           # number of productivity shocks
        self.initial_dist = initial_dist                # initial distribution of productivity
        self.pi = pi                                    # Stochastic process for employment
        self.theta = theta                              # labor income marginal tax rate
        self.eta = eta                                  # age-efficiency profile - replace with values 'ef'
        mu = np.ones(S)
        for i in range(1, S):
            mu[i] = mu[i - 1] / (1 + g_n)
        mu /= np.sum(mu)
        self.mu = mu                                    # age proportions
    def u(self, c):
        # utility of consumption
        return (c**(1-self.sigma)) / (1-self.sigma)
    def v(self, n):
        # labor disutility
        return self.psi * (n**(1+self.gamma)) / (1+self.gamma)
    def mu_c(self, c):
        # marginal utility of consumption
        return c**(-self.sigma)
    def inv_mu_c(self, value):
        # inverse marginal utility of consumption
        return value ** (-1/self.sigma)
    def inv_mu_n(self, value):
        # inverse marginal utility of labor
        return (value / self.psi) ** (1/self.gamma)
    def n_from_c(self, c, coef):
        # from FOC, coef = w * z * eta * (1 - theta)
        n = self.inv_mu_n(coef * self.mu_c(c))
        return np.minimum(n, 1)

@njit
def c_root(c, OLG, r, coef, b, constrained=False):
    n = OLG.n_from_c(c, coef)
    if constrained:
        return (1 + r) * b + coef * n - c - OLG.b_min
    else:
        return (1 + r) * b + coef * n - c

@njit
def solve_HH_SS(OLG, r=.05, w=1.05):
    # solve household problem using endogenous grid method
    b_grid = OLG.b_grid

    # initialize policy functions on grid
    b_policy = np.zeros((OLG.S, OLG.nb, OLG.nz)) # savings policy function for workers
    c_policy = np.zeros((OLG.S, OLG.nb, OLG.nz)) # consumption policy function for workers
    n_policy = np.zeros((OLG.S, OLG.nb, OLG.nz))   # labor policy function for workers


    # start at the end of life                      
    for z_index, z in enumerate(OLG.z_grid):
        coef = w * z * OLG.eta[-1] * (1 - OLG.theta)
        for b_index, b in enumerate(b_grid):
            # use root finder to solve labor-consumption problem at end of life
            args = (OLG, r, coef, b, False)
            c = brentq(c_root, 0, (1+r)*b + coef, args).root
            n = OLG.n_from_c(c, coef)
            c_policy[-1, b_index, z_index] = c
            n_policy[-1, b_index, z_index] = n
            

    # iterate backwards with Euler equation for workers
    for s in range(OLG.S-2, -1, -1):
        for z_index, z in enumerate(OLG.z_grid):
            coef = w * z * OLG.eta[s] * (1 - OLG.theta)
            # find the expected marginal utility of consumption in the next period
            E_mu_c = np.zeros(OLG.nb)
            for zp_index in range(OLG.nz):
                E_mu_c += OLG.pi[z_index, zp_index] * OLG.mu_c(c_policy[s+1, :, zp_index])
            c = OLG.inv_mu_c(OLG.beta * (1 + r) * E_mu_c)
            n = OLG.n_from_c(c, coef)
            new_grid = (b_grid + c - coef * n) / (1 + r)
            c_policy[s, :, z_index] = np.interp(b_grid, new_grid, c)
            n_policy[s, :, z_index] = OLG.n_from_c(c_policy[s, :, z_index], coef)
            b_policy[s, :, z_index] = (1+r) * b_grid + coef*n_policy[s, :, z_index] - c_policy[s, :, z_index]
            # replace policies less than b_min with b_min for borrowing constraint
            # find the first index where borrowing constraint is not binding
            last_binding = np.argmax(b_policy[s, :, z_index] >= OLG.b_min)
            b_policy[s, :last_binding, z_index] = OLG.b_min
            # loop over indicies where binding
            for i in range(last_binding):
                if n_policy[s, i, z_index] == 1:
                    # if binding and l = 1, then just use BC
                    c_policy[s, i, z_index] = (1+r) * b_grid[i] + coef - OLG.b_min
                else:
                    # essentially perform grid search here
                    args = (OLG, r, b_grid[i], coef, True)
                    # use root finder to find c that satisfies BC
                    c_policy[s, i, z_index] = brentq(c_root, 0, c_policy[s, i, z_index], args).root
                    n_policy[s, i, z_index] = OLG.n_from_c(c_policy[s, i, z_index], coef)

    return b_policy, c_policy, n_policy

@njit
def SS_distribution(OLG, b_policy):
    # estimate steady state distribution of agents with Young's method
    # find grid points around the policy functions and split probability mass

    # Initialize distribution
    h = np.zeros((OLG.S, OLG.nb, OLG.nz))

    # Initial distribution for workers
    h[0, 0, :] = OLG.initial_dist * OLG.mu[0]

    # Iterate forward for workers
    for s in range(OLG.S):
        for b_index in range(OLG.nb):
            for z_index in range(OLG.nz):
                for z_next in range(OLG.nz):
                    b_next = b_policy[s, b_index, z_index]
                    
                    # Find the position of b_next in the grid
                    if b_next <= OLG.b_min:
                        h[s+1, 0, z_next] += h[s, b_index, z_index] * OLG.pi[z_index, z_next] / (1 + OLG.g_n)
                    elif b_next >= OLG.b_max:
                        h[s+1, -1, z_next] += h[s, b_index, z_index] * OLG.pi[z_index, z_next] / (1 + OLG.g_n)
                    else:
                        # Find the two nearest grid points
                        idx = np.searchsorted(OLG.b_grid, b_next)
                        # Split the probability between the two nearest grid points
                        weight_high = (b_next - OLG.b_grid[idx-1]) / (OLG.b_grid[idx] - OLG.b_grid[idx-1])
                        weight_low = 1 - weight_high
                        h[s+1, idx-1, z_next] += weight_low * h[s, b_index, z_index] * OLG.pi[z_index, z_next] / (1 + OLG.g_n)
                        h[s+1, idx, z_next] += weight_high * h[s, b_index, z_index] * OLG.pi[z_index, z_next] / (1 + OLG.g_n)

    return h

@njit
def get_aggregates(OLG, h, n_policy):
    # compute capital and labor supply implied by household decisions
    K = 0.0
    L = 0.0
    for s in range(OLG.S):
        for b_index, b in enumerate(OLG.b_grid):
            for z_index, z in enumerate(OLG.z_grid):
                K += h[s, b_index, z_index] * b
                L += h[s, b_index, z_index] * z * OLG.eta[s] * n_policy[s, b_index, z_index]

    return K, L

@njit
def get_prices(OLG, K, L):
    # compute prices implied by capital and labor supply from firm's FOC
    r = OLG.alpha * (L / K) ** (1 - OLG.alpha) - OLG.delta
    w = (1 - OLG.alpha) * (K / L) ** OLG.alpha
    # a = OLG.theta * w * L / np.sum(OLG.mu[OLG.J_r - 1:])
    return r, w

@njit
def solve_SS(OLG, tol=0.0001, max_iter=200, rho=.1, K0=3.32, L0=0.34):
    # solve for the steady state using initial guess of capital and labor
    # initial guess K0 and L0

    K, L = K0, L0
    m = 0
    error = 100 * tol

    while error > tol:
        r, w = get_prices(OLG, K, L)
        b_policy, _, n_policy = solve_HH_SS(OLG, r, w)
        h = SS_distribution(OLG, b_policy)
        K_new, L_new = get_aggregates(OLG, h, n_policy)
        K = (1-rho) * K + rho * K_new
        L = (1-rho) * L + rho * L_new
        # print(f"K = {K}, L = {L}")
        error = max(abs(K_new - K), abs(L_new - L))
        m += 1
        if m > max_iter:
            print("No convergence")
            print("K = ", K, " L = ", L)
            break

    if m < max_iter:
        print("Converged to steady state in ", m, " iterations")
        return K, L, r, w