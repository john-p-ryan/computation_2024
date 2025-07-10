import numpy as np
from quantecon.optimize.root_finding import brentq
from scipy.interpolate import CubicSpline
from numba import njit, float64, int64
from numba.experimental import jitclass


@njit
def nonuniform_grid(a, b, n=100, density=2):
    """
    a: lower bound
    b: upper bound
    density: positive scalar for the density of the grid. > 1 for more points at the lower end
    """
    linear_points = np.linspace(0, 1, n)
    nonlinear_points = (linear_points**density) * (b - a) + a
    return nonlinear_points


primitives = [
    ("beta", float64),
    ("gamma", float64),
    ("psi", float64),
    ("sigma", float64),
    ("alpha", float64),
    ("delta", float64),
    ("N", int64),
    ("J_r", int64),
    ("n", float64),
    ("a_min", float64),
    ("a_max", float64),
    ("na", int64),
    ("a_grid", float64[:]),
    ("z_grid", float64[:]),
    ("nz", int64),
    ("initial_dist", float64[:]),
    ("pi", float64[:, :]),
    ("theta", float64),
    ("eta", float64[:]),
    ("mu", float64[:]),
]


@jitclass(primitives)
class OLGModel:
    def __init__(
        self,
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
        eta=np.ones(45),
    ):
        # Primitives
        self.beta = beta  # discount rate
        self.gamma = gamma  # Frisch elasticity
        self.psi = psi  # disutility of labor
        self.sigma = sigma  # CRRA consumption
        self.alpha = alpha  # capital share
        self.delta = delta  # depreciation rate
        self.N = N  # death age
        self.J_r = J_r  # age of retirement
        self.n = n  # population growth rate
        self.a_min = a_min  # assets lower bound
        self.a_max = a_max  # assets upper bound
        self.na = na  # number of asset grid points
        self.a_grid = nonuniform_grid(a_min, a_max, na)
        self.z_grid = z_grid  # productivity shocks
        self.nz = len(z_grid)  # number of productivity shocks
        self.initial_dist = (
            initial_dist  # initial distribution of productivity
        )
        self.pi = pi  # Stochastic process for employment
        self.theta = theta  # social security tax rate
        self.eta = eta  # age-efficiency profile - replace with values 'ef'
        mu = np.ones(N)
        for i in range(1, N):
            mu[i] = mu[i - 1] / (1 + n)
        mu /= np.sum(mu)
        self.mu = mu  # age proportions

    def u(self, c):
        return (c ** (1 - self.sigma)) / (1 - self.sigma)

    def v(self, l):
        return self.psi * (l ** (1 + self.gamma)) / (1 + self.gamma)

    def mu_c(self, c):
        return c ** (-self.sigma)

    def inv_mu_c(self, value):
        # inverse marginal utility of consumption
        return value ** (-1 / self.sigma)

    def inv_mu_l(self, value):
        # inverse marginal utility of labor
        return (value / self.psi) ** (1 / self.gamma)

    def l_from_c(self, c, coef):
        # from FOC, coef = w * z * eta * (1 - theta)
        l = self.inv_mu_l(coef * self.mu_c(c))
        return np.minimum(l, 1)


@njit
def constrained_c_root(c, OLG, r, a, coef):
    l = OLG.inv_mu_l(coef * OLG.mu_c(c))
    l = np.minimum(l, 1)
    return (1 + r) * a + coef * l - c


@njit
def HH_egm(OLG, r=0.05, w=1.05, b=0.2):
    # solve household problem using endogenous grid method
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
    c_policy_r[-1, :] = a_grid * (1 + r) + b

    # iterate backward with Euler equation
    for j in range(OLG.N - OLG.J_r - 1, -1, -1):
        c_r = OLG.inv_mu_c(OLG.beta * (1 + r) * OLG.mu_c(c_policy_r[j + 1, :]))
        # find asset holdings from BC
        new_grid = (a_grid + c_r - b) / (1 + r)
        c_policy_r[j, :] = np.interp(a_grid, new_grid, c_r)
        a_policy_r[j, :] = (1 + r) * a_grid + b - c_policy_r[j, :]

    # start just before retirement
    # find c with Euler equation, find l with FOC
    for z_index, z in enumerate(OLG.z_grid):
        coef = w * z * OLG.eta[-1] * (1 - OLG.theta)
        c_w = OLG.inv_mu_c(OLG.beta * (1 + r) * OLG.mu_c(c_policy_r[0, :]))
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
        for z_index, z in enumerate(OLG.z_grid):
            coef = w * z * OLG.eta[j] * (1 - OLG.theta)
            # find the expected marginal utility of consumption in the next period
            E_mu_c = np.zeros(OLG.na)
            for zp_index in range(OLG.nz):
                E_mu_c += OLG.pi[z_index, zp_index] * OLG.mu_c(
                    c_policy_w[j + 1, zp_index, :]
                )
            c_w = OLG.inv_mu_c(OLG.beta * (1 + r) * E_mu_c)
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
            # loop over indicies where binding
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


def policy_interpolator(OLG, c_w, c_r, l_w, a_w, a_r):
    # this takes the policy functions on the grid and returns interpolators
    c_funcs = [None] * OLG.N
    l_funcs = [None] * (OLG.J_r - 1)
    a_funcs = [None] * OLG.N
    for j in range(OLG.J_r - 1):
        c_funcs[j] = []
        l_funcs[j] = []
        a_funcs[j] = []
        for z_index in range(OLG.nz):
            c_funcs[j].append(CubicSpline(OLG.a_grid, c_w[j, z_index, :]))
            l_funcs[j].append(CubicSpline(OLG.a_grid, l_w[j, z_index, :]))
            a_funcs[j].append(CubicSpline(OLG.a_grid, a_w[j, z_index, :]))

    for j in range(OLG.J_r - 1, OLG.N):
        c_funcs[j] = CubicSpline(OLG.a_grid, c_r[j - OLG.J_r, :])
        a_funcs[j] = CubicSpline(OLG.a_grid, a_r[j - OLG.J_r, :])

    return c_funcs, l_funcs, a_funcs


@njit
def steady_dist_egm(OLG, a_policy_w, a_policy_r):
    # Initialize distributions
    h_r = np.zeros((OLG.N - OLG.J_r + 1, OLG.na))
    h_w = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))

    # Initial distribution for workers
    h_w[0, 0, :] = OLG.initial_dist * OLG.mu[0]

    # Iterate forward for workers
    for j in range(1, OLG.J_r - 1):
        for a_k in range(OLG.na):
            for z in range(OLG.nz):
                for z_next in range(OLG.nz):
                    a_next = a_policy_w[j - 1, z, a_k]

                    # Find the position of a_next in the grid
                    if a_next <= OLG.a_min:
                        h_w[j, 0, z_next] += (
                            h_w[j - 1, a_k, z]
                            * OLG.pi[z, z_next]
                            / (1 + OLG.n)
                        )
                    elif a_next >= OLG.a_max:
                        h_w[j, -1, z_next] += (
                            h_w[j - 1, a_k, z]
                            * OLG.pi[z, z_next]
                            / (1 + OLG.n)
                        )
                    else:
                        # Find the two nearest grid points
                        idx = np.searchsorted(OLG.a_grid, a_next)
                        # Split the probability between the two nearest grid points
                        weight_high = (a_next - OLG.a_grid[idx - 1]) / (
                            OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                        )
                        weight_low = 1 - weight_high
                        h_w[j, idx - 1, z_next] += (
                            weight_low
                            * h_w[j - 1, a_k, z]
                            * OLG.pi[z, z_next]
                            / (1 + OLG.n)
                        )
                        h_w[j, idx, z_next] += (
                            weight_high
                            * h_w[j - 1, a_k, z]
                            * OLG.pi[z, z_next]
                            / (1 + OLG.n)
                        )

    # Transition from workers to retirees
    for a_k in range(OLG.na):
        for z in range(OLG.nz):
            a_next = a_policy_w[-1, z, a_k]

            if a_next <= OLG.a_min:
                h_r[0, 0] += h_w[-1, a_k, z] / (1 + OLG.n)
            elif a_next >= OLG.a_max:
                h_r[0, -1] += h_w[-1, a_k, z] / (1 + OLG.n)
            else:
                idx = np.searchsorted(OLG.a_grid, a_next)
                weight_high = (a_next - OLG.a_grid[idx - 1]) / (
                    OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                )
                weight_low = 1 - weight_high
                h_r[0, idx - 1] += weight_low * h_w[-1, a_k, z] / (1 + OLG.n)
                h_r[0, idx] += weight_high * h_w[-1, a_k, z] / (1 + OLG.n)

    # Iterate forward for retirees
    for j in range(1, OLG.N - OLG.J_r + 1):
        for a_k in range(OLG.na):
            a_next = a_policy_r[j - 1, a_k]

            if a_next <= OLG.a_min:
                h_r[j, 0] += h_r[j - 1, a_k] / (1 + OLG.n)
            elif a_next >= OLG.a_max:
                h_r[j, -1] += h_r[j - 1, a_k] / (1 + OLG.n)
            else:
                idx = np.searchsorted(OLG.a_grid, a_next)
                weight_high = (a_next - OLG.a_grid[idx - 1]) / (
                    OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                )
                weight_low = 1 - weight_high
                h_r[j, idx - 1] += weight_low * h_r[j - 1, a_k] / (1 + OLG.n)
                h_r[j, idx] += weight_high * h_r[j - 1, a_k] / (1 + OLG.n)

    # Normalize the distributions
    total_mass = np.sum(h_w) + np.sum(h_r)
    h_w /= total_mass
    h_r /= total_mass

    return h_w, h_r


@njit
def markov_simulation(pi, initial_dist, T):
    """
    Simulate a Markov process.

    Parameters:
    - pi: transition matrix
    - initial_dist: initial distribution
    - T: number of periods to simulate

    Returns:
    - states: array of simulated states
    """
    states = np.zeros(T, dtype=np.int64)
    states[0] = np.searchsorted(
        (np.cumsum(initial_dist) > np.random.rand()).astype(np.int64), 1
    )

    for t in range(1, T):
        states[t] = np.searchsorted(
            (np.cumsum(pi[states[t - 1]]) > np.random.rand()).astype(np.int64),
            1,
        )

    return states


@njit
def generate_markov_shocks(OLG, M):
    """
    Generate Markov shocks for all households.

    Parameters:
    - OLG: instance of OLGModel class
    - M: number of households to simulate

    Returns:
    - productivity: array of productivity shocks for all households
    """
    productivity = np.zeros((M, OLG.J_r - 1), dtype=np.int64)

    for m in range(M):
        productivity[m] = markov_simulation(
            OLG.pi, OLG.initial_dist, OLG.J_r - 1
        )

    return productivity


@njit
def stochastic_simulation(OLG, a_policy_w, a_policy_r, productivity):
    """
    Stochastic simulation method for estimating the distribution of agents with continuous asset holdings.

    Parameters:
    - OLG: instance of OLGModel class
    - a_policy_w: asset policy function for workers (shape: [J_r-1, nz, na])
    - a_policy_r: asset policy function for retirees (shape: [N-J_r+1, na])
    - productivity: pre-generated productivity shocks (shape: [M, J_r-1])

    Returns:
    - h_w: distribution of working households (shape: [J_r-1, na, nz])
    - h_r: distribution of retired households (shape: [N-J_r+1, na])
    """
    M = productivity.shape[0]

    # Initialize arrays to store asset holdings for each household
    asset_holdings = np.zeros((M, OLG.N))

    # Simulate asset accumulation
    for m in range(M):
        for j in range(1, OLG.J_r):
            z_index = productivity[m, j - 1]
            a = asset_holdings[m, j - 1]
            # Interpolate the policy function
            asset_holdings[m, j] = np.interp(
                a, OLG.a_grid, a_policy_w[j - 1, z_index]
            )

        for j in range(OLG.J_r, OLG.N):
            a = asset_holdings[m, j - 1]
            # Interpolate the policy function
            asset_holdings[m, j] = np.interp(
                a, OLG.a_grid, a_policy_r[j - OLG.J_r]
            )

    # Initialize distribution arrays
    h_w = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))
    h_r = np.zeros((OLG.N - OLG.J_r + 1, OLG.na))

    # Bin the data and weight by age group
    for j in range(OLG.J_r - 1):
        for m in range(M):
            z_index = productivity[m, j]
            a = asset_holdings[m, j]

            if a <= OLG.a_min:
                h_w[j, 0, z_index] += OLG.mu[j]
            elif a >= OLG.a_max:
                h_w[j, -1, z_index] += OLG.mu[j]
            else:
                # Find the two nearest grid points
                idx = np.searchsorted(OLG.a_grid, a)
                # Split the probability between the two nearest grid points
                weight_high = (a - OLG.a_grid[idx - 1]) / (
                    OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                )
                weight_low = 1 - weight_high
                h_w[j, idx - 1, z_index] += weight_low * OLG.mu[j]
                h_w[j, idx, z_index] += weight_high * OLG.mu[j]

    for j in range(OLG.J_r - 1, OLG.N):
        for m in range(M):
            a = asset_holdings[m, j]

            if a <= OLG.a_min:
                h_r[j - OLG.J_r + 1, 0] += OLG.mu[j]
            elif a >= OLG.a_max:
                h_r[j - OLG.J_r + 1, -1] += OLG.mu[j]
            else:
                # Find the two nearest grid points
                idx = np.searchsorted(OLG.a_grid, a)
                # Split the probability between the two nearest grid points
                weight_high = (a - OLG.a_grid[idx - 1]) / (
                    OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                )
                weight_low = 1 - weight_high
                h_r[j - OLG.J_r + 1, idx - 1] += weight_low * OLG.mu[j]
                h_r[j - OLG.J_r + 1, idx] += weight_high * OLG.mu[j]

    # Normalize distributions
    h_w /= M
    h_r /= M

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
                L += (
                    h_w[j, a_index, z_index]
                    * z
                    * OLG.eta[j]
                    * l_w[j, z_index, a_index]
                )

    for j in range(OLG.N - OLG.J_r + 1):
        for a_index, a in enumerate(OLG.a_grid):
            K += h_r[j, a_index] * a

    return K, L


@njit
def market_clearing(
    OLG,
    tol=0.0001,
    max_iter=200,
    rho=0.02,
    K0=3.32,
    L0=0.34,
    stochastic=False,
    M=10000,
):
    # solve for the steady state using initial guess of capital and labor

    K, L = K0, L0
    n = 0
    # need initial retired share, will actually be very close to this value
    mu_r = np.sum(OLG.mu[OLG.J_r - 1 :])
    error = 100 * tol

    # Generate Markov shocks once
    if stochastic:
        productivity = generate_markov_shocks(OLG, M)

    while error > tol:
        r = OLG.alpha * (L / K) ** (1 - OLG.alpha) - OLG.delta
        w = (1 - OLG.alpha) * (K / L) ** OLG.alpha
        b = OLG.theta * w * L / mu_r
        a_w, a_r, _, _, l_w = HH_egm(OLG, r, w, b)
        if stochastic:
            h_w, h_r = stochastic_simulation(OLG, a_w, a_r, productivity)
        else:
            h_w, h_r = steady_dist_egm(OLG, a_w, a_r)
        K_new, L_new = K_L(OLG, h_w, h_r, l_w)
        K = (1 - rho) * K + rho * K_new
        L = (1 - rho) * L + rho * L_new
        # print(f"K = {K}, L = {L}")
        error = max(abs(K_new - K), abs(L_new - L))
        n += 1
        if n > max_iter:
            print("No convergence")
            print(f"K = {K}, L = {L}")
            break

    if n < max_iter:
        print(f"Converged in {n} iterations")
        return K, L, r, w, b


def stochastic_simulation_continuous(OLG, a_funcs, productivity):
    """
    Stochastic simulation method for estimating the distribution of agents using continuous policy functions.

    Parameters:
    - OLG: instance of OLGModel class
    - c_funcs: consumption policy functions (from policy_interpolator)
    - l_funcs: labor policy functions (from policy_interpolator)
    - a_funcs: asset policy functions (from policy_interpolator)
    - productivity: pre-generated productivity shocks (shape: [M, J_r-1])

    Returns:
    - h_w: distribution of working households (shape: [J_r-1, na, nz])
    - h_r: distribution of retired households (shape: [N-J_r+1, na])
    """
    M = productivity.shape[0]

    # Initialize arrays to store asset holdings for each household
    asset_holdings = np.zeros((M, OLG.N))

    # Simulate asset accumulation
    for m in range(M):
        for j in range(1, OLG.J_r):
            z_index = productivity[m, j - 1]
            a = asset_holdings[m, j - 1]
            asset_holdings[m, j] = a_funcs[j - 1][z_index](a)

        for j in range(OLG.J_r, OLG.N):
            a = asset_holdings[m, j - 1]
            asset_holdings[m, j] = a_funcs[j](a)

    # Initialize distribution arrays
    h_w = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))
    h_r = np.zeros((OLG.N - OLG.J_r + 1, OLG.na))

    # Bin the data and weight by age group
    for j in range(OLG.J_r - 1):
        for m in range(M):
            z_index = productivity[m, j]
            a = asset_holdings[m, j]

            if a <= OLG.a_min:
                h_w[j, 0, z_index] += OLG.mu[j]
            elif a >= OLG.a_max:
                h_w[j, -1, z_index] += OLG.mu[j]
            else:
                # Find the two nearest grid points
                idx = np.searchsorted(OLG.a_grid, a)
                # Split the probability between the two nearest grid points
                weight_high = (a - OLG.a_grid[idx - 1]) / (
                    OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                )
                weight_low = 1 - weight_high
                h_w[j, idx - 1, z_index] += weight_low * OLG.mu[j]
                h_w[j, idx, z_index] += weight_high * OLG.mu[j]

    for j in range(OLG.J_r - 1, OLG.N):
        for m in range(M):
            a = asset_holdings[m, j]

            if a <= OLG.a_min:
                h_r[j - OLG.J_r + 1, 0] += OLG.mu[j]
            elif a >= OLG.a_max:
                h_r[j - OLG.J_r + 1, -1] += OLG.mu[j]
            else:
                # Find the two nearest grid points
                idx = np.searchsorted(OLG.a_grid, a)
                # Split the probability between the two nearest grid points
                weight_high = (a - OLG.a_grid[idx - 1]) / (
                    OLG.a_grid[idx] - OLG.a_grid[idx - 1]
                )
                weight_low = 1 - weight_high
                h_r[j - OLG.J_r + 1, idx - 1] += weight_low * OLG.mu[j]
                h_r[j - OLG.J_r + 1, idx] += weight_high * OLG.mu[j]

    # Normalize distributions
    h_w /= M
    h_r /= M

    return h_w, h_r


def market_clearing_continuous(
    OLG, tol=0.0001, max_iter=200, rho=0.02, K0=3.32, L0=0.34, M=10000
):
    K, L = K0, L0
    n = 0
    mu_r = np.sum(OLG.mu[OLG.J_r - 1 :])
    error = 100 * tol

    # Generate Markov shocks once
    productivity = generate_markov_shocks(OLG, M)

    while error > tol:
        r = OLG.alpha * (L / K) ** (1 - OLG.alpha) - OLG.delta
        w = (1 - OLG.alpha) * (K / L) ** OLG.alpha
        b = OLG.theta * w * L / mu_r
        a_w, a_r, c_w, c_r, l_w = HH_egm(OLG, r, w, b)
        _, _, a_funcs = policy_interpolator(OLG, c_w, c_r, l_w, a_w, a_r)
        h_w, h_r = stochastic_simulation_continuous(OLG, a_funcs, productivity)
        K_new, L_new = K_L(OLG, h_w, h_r, l_w)
        K = (1 - rho) * K + rho * K_new
        L = (1 - rho) * L + rho * L_new
        error = max(abs(K_new - K), abs(L_new - L))
        n += 1
        if n > max_iter:
            print("No convergence")
            print(f"K = {K}, L = {L}")
            break

    if n < max_iter:
        print(f"Converged in {n} iterations")
        return K, L, r, w, b
