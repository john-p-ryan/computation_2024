import numpy as np
from numba import jit, njit
from numba import float64, int64
from numba.experimental import jitclass
from quantecon.optimize.root_finding import brentq

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
    ("a_grid_density", float64),
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
        a_max=30.0,
        na=1000,
        a_grid_density=1.5,
        z_grid=np.array([2.0, 0.3]),
        initial_dist=np.array([0.2037, 0.7963]),
        pi=np.array([[0.9261, 0.0739], [0.0189, 0.9811]]),
        theta=0.11,
        eta=np.ones(45),
    ):
        # Primitives
        self.beta = beta  # discount rate
        self.gamma = gamma  # labor supply elasticity parameter
        self.psi = psi  # labor disutility level
        self.sigma = sigma  # CRRA consumption
        self.alpha = alpha  # capital share
        self.delta = delta  # depreciation rate
        self.N = N  # death age
        self.J_r = J_r  # age of retirement
        self.n = n  # population growth rate
        self.a_min = a_min  # assets lower bound
        self.a_max = a_max  # assets upper bound
        self.na = na  # number of asset grid points
        uniform_grid = np.linspace(0, 1, na)
        # Apply a nonlinear transformation to change density of points
        self.a_grid = a_min + (a_max - a_min) * (
            uniform_grid**a_grid_density
        )  # asset grid
        self.z_grid = z_grid  # productivity shocks
        self.nz = len(z_grid)  # number of productivity shocks
        self.initial_dist = (
            initial_dist  # initial distribution of productivity
        )
        self.pi = pi  # Stochastic process for employment
        self.theta = theta  # social security tax rate
        self.eta = eta  # replace with the actual values of 'ef'
        mu = np.ones(N)
        for i in range(1, N):
            mu[i] = mu[i - 1] / (1 + n)
        mu /= np.sum(mu)
        self.mu = mu

    def u(self, c):
        return c ** (1 - self.sigma) / (1 - self.sigma)

    def v(self, l):
        return self.psi * l ** (1 + self.gamma) / (1 + self.gamma)

    def U(self, c, l):
        return self.u(c) - self.v(l)


@njit
def find_l(l, OLG, coef, r, a, ap):
    # zero of this function gives l as a function of a and a'
    return (
        (1 + r) * a
        - ap
        + coef * l
        - (OLG.psi * l**OLG.gamma / coef) ** (-1 / OLG.sigma)
    )


@njit
def V_induction(OLG, r=0.05, w=1.05, b=0.2):
    # OLG is an instance of OLGModel

    # initialize value functions, policy functions and utility functions
    V_r = np.zeros((OLG.N - OLG.J_r + 1, OLG.na))
    g_r = np.zeros((OLG.N - OLG.J_r + 1, OLG.na))
    c_r = np.zeros((OLG.N - OLG.J_r + 1, OLG.na))
    V_w = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))
    g_w = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))
    l_w = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))
    c_w = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))

    # initialize with age N
    V_r[-1, :] = OLG.u((1 + r) * OLG.a_grid + b)
    c_r[-1, :] = (1 + r) * OLG.a_grid + b

    # Precompute budgets for each a
    budgets = (1 + r) * OLG.a_grid + b

    # Value function induction for retired first
    for j in range(
        OLG.N - OLG.J_r - 1, -1, -1
    ):  # age = N-1, iterate backwards
        for a_index, a in enumerate(OLG.a_grid):
            budget = budgets[a_index]  # budget

            # Vectorized computation of c and val
            c = budget - OLG.a_grid
            valid = c > 0
            val = np.where(valid, OLG.u(c) + OLG.beta * V_r[j + 1, :], -np.inf)

            # Find the maximum value and its index
            ap_index = np.argmax(val)
            candidate_max = val[ap_index]

            g_r[j, a_index] = OLG.a_grid[ap_index]  # update policy function
            c_r[j, a_index] = c[ap_index]  # update consumption
            V_r[j, a_index] = candidate_max  # update value function

    # initialize with age J_r
    for z_index, z in enumerate(OLG.z_grid):
        coef = w * z * OLG.eta[-1] * (1 - OLG.theta)

        for a_index, a in enumerate(OLG.a_grid):
            candidate_max = -np.inf  # bad candidate max

            # grid search
            for ap_index, ap in enumerate(OLG.a_grid):
                args = (OLG, coef, r, a, ap)
                l = brentq(find_l, 0, 1000, args).root  # labor supply given a'
                l = max(0, min(1, l))  # enforce bounds
                c = (1 + r) * a + coef * l - ap  # consumption given a'
                if c > 0:  # check for positivity
                    val = (
                        OLG.U(c, l) + OLG.beta * V_r[0, ap_index]
                    )  # compute value function
                    if val > candidate_max:  # check for new max value
                        candidate_max = val  # update max value
                        g_w[-1, a_index, z_index] = (
                            ap  # update policy function
                        )
                        l_w[-1, a_index, z_index] = l  # update labor supply
                        c_w[-1, a_index, z_index] = c  # update consumption

            V_w[-1, a_index, z_index] = candidate_max  # update value function

    # Value function induction for workers
    for j in range(OLG.J_r - 3, -1, -1):  # age = J_r-2, iterate backwards
        for z_index, z in enumerate(OLG.z_grid):
            coef = w * z * OLG.eta[j] * (1 - OLG.theta)

            for a_index, a in enumerate(OLG.a_grid):
                candidate_max = -np.inf  # bad candidate max

                # grid search
                for ap_index, ap in enumerate(
                    OLG.a_grid
                ):  # loop over vals of a'
                    args = (OLG, coef, r, a, ap)
                    l = brentq(
                        find_l, 0, 1000, args
                    ).root  # labor supply given a'
                    l = max(0, min(1, l))  # enforce bounds
                    c = (1 + r) * a + coef * l - ap  # consumption given a'

                    if c > 0:  # check for positivity
                        pi_next = np.ascontiguousarray(OLG.pi[z_index, :])
                        V_next = np.ascontiguousarray(V_w[j + 1, ap_index, :])
                        val = OLG.U(c, l) + OLG.beta * np.dot(
                            pi_next, V_next
                        )  # compute value function
                        if val > candidate_max:  # check for new max value
                            candidate_max = val  # update max value
                            g_w[j, a_index, z_index] = (
                                ap  # update policy function
                            )
                            l_w[j, a_index, z_index] = l  # update labor supply
                            c_w[j, a_index, z_index] = c  # update consumption

                V_w[j, a_index, z_index] = (
                    candidate_max  # update value function
                )
    return V_r, g_r, c_r, V_w, g_w, c_w, l_w


@njit
def steady_dist(OLG, g_w, g_r):
    h_r = np.zeros((OLG.N - OLG.J_r + 1, OLG.na))
    h_w = np.zeros((OLG.J_r - 1, OLG.na, OLG.nz))

    # take initial dist of productivity and age 1
    h_w[0, 0, :] = OLG.initial_dist * OLG.mu[0]

    # iterate F forward through age using policy functions for workers
    for j in range(1, OLG.J_r - 1):
        for z_index in range(OLG.nz):
            for a_index, a in enumerate(OLG.a_grid):
                for zp_index in range(OLG.nz):
                    for ap_index, ap in enumerate(OLG.a_grid):
                        if ap == g_w[j - 1, a_index, z_index]:
                            h_w[j, ap_index, zp_index] += (
                                h_w[j - 1, a_index, z_index]
                                * OLG.pi[z_index, zp_index]
                                / (1 + OLG.n)
                            )

    # take dist of initial retired from last period of employed
    for a_index, a in enumerate(OLG.a_grid):
        for ap_index, ap in enumerate(OLG.a_grid):
            for z_index in range(OLG.nz):
                if ap == g_w[OLG.J_r - 2, a_index, z_index]:
                    h_r[0, a_index] += h_w[-1, a_index, z_index] / (1 + OLG.n)

    # iterate F forward through age using policy functions for retired
    for j in range(1, OLG.N - OLG.J_r):
        for a_index, a in enumerate(OLG.a_grid):
            for ap_index, ap in enumerate(OLG.a_grid):
                if ap == g_r[j - 1, a_index]:
                    h_r[j, ap_index] += h_r[j - 1, a_index] / (1 + OLG.n)

    # renormalize to reduce numerical error
    denominator = np.sum(h_r) + np.sum(h_w)
    h_w /= denominator
    h_r /= denominator
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
                    * l_w[j, a_index, z_index]
                )

    for j in range(OLG.N - OLG.J_r):
        for a_index, a in enumerate(OLG.a_grid):
            K += h_r[j, a_index] * a

    return K, L


@njit
def market_clearing(OLG, tol=0.0001, max_iter=200, rho=0.02, K0=3.32, L0=0.34):
    # solve for the steady state using initial guess of capital and labor

    K, L = K0, L0
    n = 0
    # need initial retired share, will actually be very close to this value
    mu_r = np.sum(OLG.mu[OLG.J_r - 1 :])
    error = 100 * tol
    while error > tol:
        r = OLG.alpha * (L / K) ** (1 - OLG.alpha) - OLG.delta
        w = (1 - OLG.alpha) * (K / L) ** OLG.alpha
        b = OLG.theta * w * L / mu_r
        _, g_r, _, _, g_w, _, l_w = V_induction(OLG, r, w, b)
        h_w, h_r = steady_dist(OLG, g_w, g_r)
        K_new, L_new = K_L(OLG, h_w, h_r, l_w)
        K = (1 - rho) * K + rho * K_new
        L = (1 - rho) * L + rho * L_new
        # print(f"K = {K}, L = {L}")
        error = max(abs(K_new - K), abs(L_new - L))
        n += 1
        if n > max_iter:
            print("No convergence")
            break
    print(f"K = {K}, L = {L}")
    if n < max_iter:
        print(f"Converged in {n} iterations")
        return K, L, r, w, b
