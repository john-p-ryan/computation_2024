import numpy as np
from scipy.interpolate import CubicSpline
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
    ('N', int64),
    ('J_r', int64),
    ('n', float64),
    ('a_min', float64),
    ('a_max', float64),
    ('na', int64),
    ('a_grid', float64[:]),
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
                 N=66, 
                 J_r=46, 
                 n=0.011,
                 a_min=0.00001,
                 a_max=35.0, 
                 na=1000, 
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
        self.N = N                                      # death age
        self.J_r = J_r                                  # age of retirement
        self.n = n                                      # population growth rate
        self.a_min = a_min                              # assets lower bound
        self.a_max = a_max                              # assets upper bound
        self.na = na                                    # number of asset grid points
        self.a_grid = nonuniform_grid(a_min, a_max, na)
        self.z_grid = z_grid                            # productivity shocks
        self.nz = len(z_grid)                           # number of productivity shocks
        self.initial_dist = initial_dist                # initial distribution of productivity
        self.pi = pi                                    # Stochastic process for employment
        self.theta = theta                              # social security tax rate
        self.eta = eta                                  # age-efficiency profile - replace with values 'ef'
        mu = np.ones(N)
        for i in range(1, N):
            mu[i] = mu[i - 1] / (1 + n)
        mu /= np.sum(mu)
        self.mu = mu                                    # age proportions
    def u(self, c):
        return (c**(1-self.sigma)) / (1-self.sigma)
    def v(self, l):
        return self.psi * (l**(1+self.gamma)) / (1+self.gamma)
    def mu_c(self, c):
        return c**(-self.sigma)
    def inv_mu_c(self, value):
        # inverse marginal utility of consumption
        return value ** (-1/self.sigma)
    def l_from_c(self, c, coef):
        # from FOC, coef = w * z * eta * (1 - theta)
        return (coef * c ** (-self.sigma) / self.psi) ** (1/self.gamma)

def HH_egm(OLG, r=.05, w=1.05, b=.2):
    # solve household problem using endogenous grid method
    c_funcs = [None] * OLG.N
    a_grid = OLG.a_grid

    '''
    # find the upper bounds for assets at each working age
    a_upper = 0.0
    for j in range(OLG.J_r - 1):
        period_income = w * OLG.z_grid[0] * OLG.eta[j] * (1 - OLG.theta)
        a_upper = period_income + a_upper * (1 + r)                     # maximum assets assuming maximum income and savings
        a_grids[j, :] = nonuniform_grid(OLG.a_min, a_upper, OLG.na)
    
    for j in range(OLG.J_r - 1, OLG.N):
        a_grids[j, :] = nonuniform_grid(OLG.a_min, a_upper, OLG.na)
    '''


    # start at the end of life                      
    c_r = a_grid * (1 + r) + b           # retired consumption policy
    c_funcs[-1] = CubicSpline(a_grid, c_r) 
    
    # iterate backward with Euler equation
    for j in range(OLG.N - 1, OLG.J_r - 2, -1):
        c_next = c_funcs[j](a_grid)
        c_r = OLG.inv_mu_c(OLG.beta * (1+r) * OLG.mu_c(c_next))
        # find asset holdings from BC
        new_grid = (a_grid + c_r - b) / (1 + r)
        c_funcs[j - 1] = CubicSpline(new_grid, c_r)

    # start just before retirement
    c_funcs[OLG.J_r - 2] = []
    # find c with Euler equation, find l with FOC
    for z_index, z in enumerate(OLG.z_grid):
        coef = w * z * OLG.eta[-1] * (1 - OLG.theta)
        c_next = c_funcs[OLG.J_r - 1](a_grid)
        c_w = OLG.inv_mu_c(OLG.beta * (1 + r) * OLG.mu_c(c_next))
        l_w = OLG.l_from_c(c_w, coef)
        l_w = np.clip(l_w, 0, 1) # enforce bounds
        # find asset holdings from BC
        new_grid = (a_grid + c_w - coef * l_w) / (1 + r)
        # get rid of the values where new_grid is negative
        negative_indices = np.where(new_grid < 0)
        new_grid = np.delete(new_grid, negative_indices)
        c_w = np.delete(c_w, negative_indices)
        c_funcs[OLG.J_r - 2].append(CubicSpline(new_grid, c_w))

    # iterate backwards with Euler equation for workers
    for j in range(OLG.J_r - 2, 0, -1):
        c_funcs[j - 1] = []
        for z_index, z in enumerate(OLG.z_grid):
            coef = w * z * OLG.eta[j-1] * (1 - OLG.theta)
            # find the expected marginal utility of consumption in the next period
            E_mu_c = np.zeros(OLG.na)
            for zp_index in range(OLG.nz):
                mu_c_zp = OLG.mu_c(c_funcs[j][zp_index](a_grid))
                E_mu_c += OLG.pi[z_index, zp_index] * mu_c_zp
            c_w = OLG.inv_mu_c(OLG.beta * (1 + r) * E_mu_c)
            l_w = OLG.l_from_c(c_w, coef)
            l_w = np.clip(l_w, 0, 1) # enforce bounds
            new_grid = (a_grid + c_w - coef * l_w) / (1 + r)
            negative_indices = np.where(new_grid < OLG.a_min)
            new_grid = np.delete(new_grid, negative_indices)
            c_w = np.delete(c_w, negative_indices)
            l_w = np.delete(l_w, negative_indices)
            c_funcs[j - 1].append(CubicSpline(new_grid, c_w))
    
    # get policy functions on grid using c_funcs
    a_policy_w = np.empty((OLG.J_r-1, OLG.nz, OLG.na)) # savings policy function for workers
    a_policy_r = np.empty((OLG.N-OLG.J_r, OLG.na))         # savings policy function for retirees
    c_policy_w = np.empty((OLG.J_r-1, OLG.nz, OLG.na)) # consumption policy function for workers
    c_policy_r = np.empty((OLG.N-OLG.J_r, OLG.na))         # consumption policy function for retirees
    l_policy = np.empty((OLG.J_r-1, OLG.nz, OLG.na))   # labor policy function for workers

    for j in range(OLG.J_r - 1):
        for z_index, z in enumerate(OLG.z_grid):
            coef = w * z * OLG.eta[j] * (1 - OLG.theta)
            c_policy_w[j, z_index, :] = c_funcs[j][z_index](a_grid)
            l_policy[j, z_index, :] = np.clip(OLG.l_from_c(c_policy_w[j, z_index, :], coef), 0, 1)
            a_policy_w[j, z_index, :] = (1+r) * a_grid + coef*l_policy[j, z_index, :] - c_policy_w[j, z_index, :]

    for j in range(OLG.N - OLG.J_r):
        c_policy_r[j, :] = c_funcs[j + OLG.J_r](a_grid)
        a_policy_r[j, :] = (1+r) * a_grid + b - c_policy_r[j, :]

    return a_policy_r, a_policy_w, c_policy_r, c_policy_w, l_policy

@njit
def steady_dist_egm(OLG, a_policy_r, a_policy_w):
    # Initialize distributions
    h_r = np.zeros((OLG.N - OLG.J_r, OLG.na))
    h_w = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))

    # Initial distribution for workers
    h_w[0, 0, :] = OLG.initial_dist * OLG.mu[0]

    # Iterate forward for workers
    for j in range(1, OLG.J_r - 1):
        for a_k in range(OLG.na):
            for z in range(OLG.nz):
                for z_next in range(OLG.nz):
                    a_next = a_policy_w[j-1, z, a_k]
                    
                    # Find the position of a_next in the grid
                    if a_next <= OLG.a_grid[0]:
                        h_w[j, 0, z_next] += h_w[j-1, a_k, z] * OLG.pi[z, z_next] / (1 + OLG.n)
                    elif a_next >= OLG.a_grid[-1]:
                        h_w[j, -1, z_next] += h_w[j-1, a_k, z] * OLG.pi[z, z_next] / (1 + OLG.n)
                    else:
                        # Find the two nearest grid points
                        idx = np.searchsorted(OLG.a_grid, a_next)
                        # Split the probability between the two nearest grid points
                        weight_high = (a_next - OLG.a_grid[idx-1]) / (OLG.a_grid[idx] - OLG.a_grid[idx-1])
                        weight_low = 1 - weight_high
                        h_w[j, idx-1, z_next] += weight_low * h_w[j-1, a_k, z] * OLG.pi[z, z_next] / (1 + OLG.n)
                        h_w[j, idx, z_next] += weight_high * h_w[j-1, a_k, z] * OLG.pi[z, z_next] / (1 + OLG.n)

    # Transition from workers to retirees
    for a_k in range(OLG.na):
        for z in range(OLG.nz):
            a_next = a_policy_w[OLG.J_r-2, z, a_k]
            
            if a_next <= OLG.a_grid[0]:
                h_r[0, 0] += h_w[OLG.J_r-2, a_k, z] / (1 + OLG.n)
            elif a_next >= OLG.a_grid[-1]:
                h_r[0, -1] += h_w[OLG.J_r-2, a_k, z] / (1 + OLG.n)
            else:
                idx = np.searchsorted(OLG.a_grid, a_next)
                weight_high = (a_next - OLG.a_grid[idx-1]) / (OLG.a_grid[idx] - OLG.a_grid[idx-1])
                weight_low = 1 - weight_high
                h_r[0, idx-1] += weight_low * h_w[OLG.J_r-2, a_k, z] / (1 + OLG.n)
                h_r[0, idx] += weight_high * h_w[OLG.J_r-2, a_k, z] / (1 + OLG.n)

    # Iterate forward for retirees
    for j in range(1, OLG.N - OLG.J_r):
        for a_k in range(OLG.na):
            a_next = a_policy_r[j-1, a_k]
            
            if a_next <= OLG.a_grid[0]:
                h_r[j, 0] += h_r[j-1, a_k] / (1 + OLG.n)
            elif a_next >= OLG.a_grid[-1]:
                h_r[j, -1] += h_r[j-1, a_k] / (1 + OLG.n)
            else:
                idx = np.searchsorted(OLG.a_grid, a_next)
                weight_high = (a_next - OLG.a_grid[idx-1]) / (OLG.a_grid[idx] - OLG.a_grid[idx-1])
                weight_low = 1 - weight_high
                h_r[j, idx-1] += weight_low * h_r[j-1, a_k] / (1 + OLG.n)
                h_r[j, idx] += weight_high * h_r[j-1, a_k] / (1 + OLG.n)

    # Normalize the distributions
    total_mass = np.sum(h_w) + np.sum(h_r)
    h_w /= total_mass
    h_r /= total_mass

    return h_w, h_r

@njit
def K_L(OLG, h_w, h_r, l_w):
    # compute capital and labor supply implied by household decisions
    K = 0.0
    L = 0.0
    for j in range(OLG.J_r - 1):  # workers
        for a_index, a in enumerate(OLG.a_grid):
            for z_index, z in enumerate(OLG.z_grid):
                K += h_w[j, a_index, z_index] * a
                L += h_w[j, a_index, z_index] * z * OLG.eta[j] * l_w[j, a_index, z_index]

    for j in range(OLG.N - OLG.J_r):
        for a_index, a in enumerate(OLG.a_grid):
            K += h_r[j, a_index] * a

    return K, L

def market_clearing(OLG, tol=0.0001, max_iter=200, rho=.02, K0=3.32, L0=0.34):
    # solve for the steady state using initial guess of capital and labor

    K, L = K0, L0
    n = 0
    # need initial retired share, will actually be very close to this value
    mu_r = np.sum(OLG.mu[OLG.J_r - 1:])
    error = 100 * tol
    while error > tol:
        r = OLG.alpha * (L / K) ** (1 - OLG.alpha) - OLG.delta
        w = (1 - OLG.alpha) * (K / L) ** OLG.alpha
        b = OLG.theta * w * L / mu_r
        _, g_r, _, _, g_w, _, l_w = V_induction(OLG, r, w, b)
        F_r, F_w = steady_dist(OLG, g_r, g_w)
        K_new, L_new = K_L(OLG, F_w, F_r, l_w)
        K = (1-rho) * K + rho * K_new
        L = (1-rho) * L + rho * L_new
        print(f"K = {K}, L = {L}")
        error = max(abs(K_new - K), abs(L_new - L))
        n += 1
        if n > max_iter:
            print("No convergence")
            break

    if n < max_iter:
        print(f"Converged in {n} iterations")
        return K, L, r, w, b