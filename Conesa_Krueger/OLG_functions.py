import numpy as np

class OLGModel:
    def __init__(self, 
                 beta=0.97, 
                 gamma=0.42, 
                 sigma=2.0, 
                 alpha=0.36, 
                 delta=0.06, 
                 N=66, 
                 J_r=46, 
                 n=0.011,
                 a_min=0.001, 
                 a_max=30.0, 
                 na=1000, 
                 z_grid=np.array([3.0, 0.5]), 
                 initial_dist=np.array([0.2037, 0.7963]),
                 pi=np.array([[0.9261, 0.0739], [0.0189, 0.9811]]),
                 theta=0.11, 
                 eta=np.ones(45)):
        # Primitives
        self.beta = beta            # discount rate
        self.gamma = gamma          # Cobb Douglas consumption weight
        self.sigma = sigma          # CRRA
        self.alpha = alpha          # capital share
        self.delta = delta          # depreciation rate
        self.N = N                  # death age
        self.J_r = J_r              # age of retirement
        self.n = n                  # population growth rate
        self.a_min = a_min          # assets lower bound
        self.a_max = a_max          # assets upper bound
        self.na = na                # number of asset grid points
        self.a_grid = np.linspace(a_min, a_max, na)     # asset grid
        self.z_grid = z_grid                            # productivity shocks
        self.nz = len(z_grid)                           # number of productivity shocks
        self.initial_dist = initial_dist                # initial distribution of productivity
        self.pi = pi                                    # Stochastic process for employment
        self.theta = theta                              # social security tax rate
        self.eta = eta                                  # replace with the actual values of 'ef'
        mu = np.ones(N)
        for i in range(1, N):
            mu[i] = mu[i - 1] / (1 + n)
        mu /= np.sum(mu)
        self.mu = mu

    def u(self, c, l):
        return (c**self.γ * (1-l)**(1-self.γ))**(1-self.σ) / (1-self.σ)

    def V_induction(self, r=.05, w=1.05, b=.2):
        # initialize value functions, policy functions and utility functions
        V_r = np.zeros((self.N - self.J_r, self.na))
        g_r = np.zeros((self.N - self.J_r, self.na))
        u_r = np.zeros((self.N - self.J_r, self.na))
        V_w = np.zeros((self.J_r - 1, self.na, self.nz))
        g_w = np.zeros((self.J_r - 1, self.na, self.nz))
        u_w = np.zeros((self.J_r - 1, self.na, self.nz))
        l_w = np.zeros((self.J_r - 1, self.na, self.nz))

        # initialize with age N
        for a_index, a in enumerate(self.a_grid):
            V_r[-1, a_index] = self.u((1+r)*a + b, 0)
            u_r[-1, a_index] = self.u((1+r)*a + b, 0)


        # Value function induction for retired first
        for j in range(self.N - self.Jʳ - 2, -1, -1):  # age = N-1, iterate backwards
            for a_index, a in enumerate(self.a_grid):
                candidate_max = -np.inf  # bad candidate max
                budget = (1 + r) * a + b  # budget

                # perform grid search for a'
                for ap_index, ap in enumerate(self.a_grid):     # loop over possible selections of a'
                    c = budget - ap                             # consumption given a' selection
                    if c > 0:                                  # check for positivity
                        val = self.u(c, 0) + self.beta * V_r[j + 1, ap_index]  # compute value function
                        if val > candidate_max:                                 # check for new max value
                            candidate_max = val                                 # update max value
                            g_r[j, a_index] = ap                                # update policy function
                            u_r[j, a_index] = self.u(c, 0)                      # update utility function

                V_r[j, a_index] = candidate_max  # update value function

        # initialize with age J_r
        for z_index, z in enumerate(self.z_grid):
            e = z * self.eta[-1]

            for a_index, a in enumerate(self.a_grid):
                candidate_max = -np.inf  # bad candidate max

                # grid search
                for ap_index, ap in enumerate(self.a_grid):
                    l = (self.gamma*(1 - self.theta)*e*w - (1-self.gamma)*(a*(1+r)-ap)) / ((1-self.theta)*w*e)  # labor supply given a'
                    l = max(0, min(1, l))  # enforce bounds
                    c = (1+r)*a + w*e*(1-self.theta)*l - ap  # consumption given a'
                    if c > 0:  # check for positivity
                        val = self.u(c, l) + self.beta * V_r[0, ap_index]  # compute value function
                        if val > candidate_max:  # check for new max value
                            candidate_max = val  # update max value
                            g_w[-1, a_index, z_index] = ap  # update policy function
                            l_w[-1, a_index, z_index] = l  # update labor supply
                            u_w[-1, a_index, z_index] = self.u(c, l)  # update utility function

                V_w[-1, a_index, z_index] = candidate_max  # update value function

        # Value function induction for workers
        for j in range(self.Jʳ - 2, -1, -1):  # age
            for z_index, z in enumerate(self.z_grid):
                e = z * self.eta[j]

                for a_index, a in enumerate(self.a_grid):
                    candidate_max = -np.inf  # bad candidate max

                    # grid search
                    for ap_index, ap in enumerate(self.a_grid):  # loop over vals of a'
                        l = (self.gamma*(1-self.theta)*e*w - (1-self.gamma)*(a*(1+r)-ap)) / ((1-self.theta)*w*e)  # labor supply given a'
                        l = max(0, min(1, l))
                        c = (1+r) * a + w * e * (1 - self.theta) * l - ap  # consumption given a'

                        if c > 0:  # check for positivity
                            val = self.u(c, l) + self.β * np.dot(self.π[z_index], self.Vʷ[j + 1, ap_index])  # compute value function
                            if val > candidate_max:  # check for new max value
                                candidate_max = val  # update max value
                                g_w[j, a_index, z_index] = ap  # update policy function
                                l_w[j, a_index, z_index] = l  # update labor supply
                                u_w[j, a_index, z_index] = self.u(c, l)  # update utility function

                    V_w[j, a_index, z_index] = candidate_max  # update value function
        return V_r, g_r, u_r, V_w, g_w, u_w, l_w

    def steady_dist(self, g_r, g_w):
        F_r = np.zeros((self.N - self.J_r, self.na))
        F_w = np.zeros((self.J_r - 1, self.na, self.nz))

        # take initial dist of productivity and age 1
        F_w[0, 0, :] = self.initial_dist * self.mu[0]

        # iterate F forward through age using policy functions for workers
        for j in range(1, self.J_r - 1):
            for z_index in range(self.nz):
                for a_index, a in enumerate(self.a_grid):
                    for zp_index in range(self.nz):
                        for ap_index, ap in enumerate(self.a_grid):
                            if ap == g_w[j - 1, a_index, z_index]:
                                F_w[j, ap_index, zp_index] += F_w[j - 1, a_index, z_index] * self.pi[z_index, zp_index] / (1 + self.n)

        # take dist of initial retired from last period of employed
        for a_index, a in enumerate(self.a_grid):
            for ap_index, ap in enumerate(self.a_grid):
                for z_index in range(self.nz):
                    if ap == g_w[self.J_r - 2, a_index, z_index]:
                        F_w[0, a_index] += F_w[self.J_r - 2, a_index, z_index] / (1 + self.n)

        # iterate F forward through age using policy functions for retired
        for j in range(1, self.N - self.J_r):
            for a_index, a in enumerate(self.a_grid):
                for ap_index, ap in enumerate(self.a_grid):
                    if ap == g_r[j - 1, a_index]:
                        F_r[j, ap_index] += F_r[j - 1, a_index] / (1 + self.n)

        # renormalize to reduce numerical error
        denominator = np.sum(F_r) + np.sum(F_w)
        F_r /= denominator
        F_w /= denominator
        return F_r, F_w

    def K_L(self, F_w, F_r, l_w):
        # compute capital and labor supply implied by household decisions
        K = 0.0
        L = 0.0
        for j in range(self.J_r - 1):  # workers
            for a_index, a in enumerate(self.a_grid):
                for z_index, z in enumerate(self.z_grid):
                    K += F_w[j, a_index, z_index] * a
                    L += F_w[j, a_index, z_index] * z * self.eta[j] * l_w[j, a_index, z_index]

        for j in range(self.N - self.J_r):
            for a_index, a in enumerate(self.a_grid):
                K += F_r[j, a_index] * a

        return K, L

    def market_clearing(self, tol=0.0001, max_iter=200, rho=.02, K0=3.32, L0=0.34):
        # solve for the steady state using initial guess of capital and labor

        K, L = K0, L0
        n = 0
        # need initial retired share, will actually be very close to this value
        mu_r = np.sum(self.mu[self.J_r - 1:])
        error = 100 * tol
        while error > tol:
            r = self.alpha * (L / K) ** (1 - self.alpha) - self.delta
            w = (1 - self.alpha) * (K / L) ** self.alpha
            b = self.theta * w * L / mu_r
            _, g_r, _, _, g_w, _, l_w = self.V_induction(r, w, b)
            F_r, F_w = self.steady_dist(g_r, g_w)
            K_new, L_new = self.K_L(F_w, F_r, l_w)
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