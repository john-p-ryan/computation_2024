#%%
import numpy as np
import scipy.optimize as opt
import pytest
from ogcore import tax, utils
from ogcore.parameters import Specifications
import household_stoch as household


# %%
test_data = [
    (0.1, 1, 10),
    (0.2, 2.5, 55.90169944),
    (
        np.array([0.5, 6.2, 1.5]),
        3.2,
        np.array([9.18958684, 0.002913041, 0.273217159]),
    ),
]


@pytest.mark.parametrize(
    "c,sigma,expected", test_data, ids=["Scalar 0", "Scalar 1", "Vector"]
)
def test_marg_ut_cons(c, sigma, expected):
    # Test marginal utility of consumption calculation
    test_value = household.marg_ut_cons(c, sigma)

    assert np.allclose(test_value, expected)



# %%

# Tuples in order: n, p, expected result
p1 = Specifications()
p1.b_ellipse = 0.527
p1.upsilon = 1.497
p1.ltilde = 1.0
p1.chi_n = 3.3

p2 = Specifications()
p2.b_ellipse = 0.527
p2.upsilon = 0.9
p2.ltilde = 1.0
p2.chi_n = 3.3

p3 = Specifications()
p3.b_ellipse = 0.527
p3.upsilon = 0.9
p3.ltilde = 2.3
p3.chi_n = 3.3

p4 = Specifications()
p4.b_ellipse = 2.6
p4.upsilon = 1.497
p4.ltilde = 1.0
p4.chi_n = 3.3

test_data = [
    (0.87, p1, 2.825570309),
    (0.0, p1, 0.0009117852028298067),
    (0.99999, p1, 69.52423604),
    (0.00001, p1, 0.005692782),
    (0.8, p2, 1.471592068),
    (0.8, p3, 0.795937549),
    (0.8, p4, 11.66354267),
    (
        np.array([[0.8, 0.9, 0.3], [0.5, 0.2, 0.99]]),
        p1,
        np.array(
            [
                [2.364110379, 3.126796062, 1.014935377],
                [1.4248841, 0.806333875, 6.987729463],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "n,params,expected",
    test_data,
    ids=["1", "2", "3", "4", "5", "6", "7", "8"],
)
def test_marg_ut_labor(n, params, expected):
    # Test marginal utility of labor calculation
    test_value = household.marg_ut_labor(n, params.chi_n, params)

    assert np.allclose(test_value, expected)


#%%

# Test cases for inv_mu_c are the reverse of the test cases for
# marg_ut_cons from test_household.py
inv_mu_c_test_data = [
    (10, 1, 0.1),
    (55.90169944, 2.5, 0.2),
    (
        np.array([9.18958684, 0.002913041, 0.273217159]),
        3.2,
        np.array([0.5, 6.2, 1.5]),
    ),
]


@pytest.mark.parametrize(
    "value,sigma,expected",
    inv_mu_c_test_data,
    ids=["Scalar 1", "Scalar 2", "Vector"],
)
def test_inv_mu_c(value, sigma, expected):
    """
    Test the inverse marginal utility of consumption function `inv_mu_c`.
    """
    test_value = household.inv_mu_c(value, sigma)
    assert np.allclose(test_value, expected)



#%%
# Setup for marg_ut_beq tests
p1 = Specifications()
p1.chi_b = np.array([1.5, 2.5, 5.0])

# Case 1: Scalar, unconstrained b
b1 = 2.0
sigma1 = 2.0
j1 = 1
expected1 = p1.chi_b[j1] * (b1**-sigma1)

# Case 2: Scalar, constrained b (b < epsilon)
b2 = 0.00005
sigma2 = 2.0
j2 = 0
epsilon = 0.0001
# Note: The calculation for the constrained part is independent of chi_b
# as currently implemented.
b2_quad = (-sigma2 * (epsilon ** (-sigma2 - 1))) / 2
b1_quad = (epsilon**-sigma2) - 2 * b2_quad * epsilon
expected2 = 2 * b2_quad * b2 + b1_quad

# Case 3: Vector, mixed constrained and unconstrained
b3 = np.array([2.0, 0.00005])
sigma3 = 2.0
j3 = 1
# The first element is unconstrained, the second is constrained
expected3 = np.array([p1.chi_b[j3] * (b3[0] ** -sigma3), expected2])

# Case 4: Vector, all unconstrained
b4 = np.array([2.0, 3.0])
sigma4 = 1.5
j4 = 2
expected4 = p1.chi_b[j4] * (b4**-sigma4)

marg_ut_beq_test_data = [
    (b1, sigma1, j1, p1, expected1),
    (b2, sigma2, j2, p1, expected2),
    (b3, sigma3, j3, p1, expected3),
    (b4, sigma4, j4, p1, expected4),
]


@pytest.mark.parametrize(
    "b,sigma,j,p,expected",
    marg_ut_beq_test_data,
    ids=[
        "Scalar, unconstrained",
        "Scalar, constrained",
        "Vector, mixed",
        "Vector, unconstrained",
    ],
)
def test_marg_ut_beq(b, sigma, j, p, expected):
    """
    Test the marginal utility of bequests function `marg_ut_beq`.
    """
    test_value = household.marg_ut_beq(b, sigma, j, p)
    assert np.allclose(test_value, expected)


# %%

p1 = Specifications()
p1.zeta = np.array([[0.1, 0.3], [0.15, 0.4], [0.05, 0.0]])
p1.S = 3
p1.J = 2
p1.T = 3
p1.lambdas = np.array([0.6, 0.4])
p1.omega_SS = np.array([0.25, 0.25, 0.5])
p1.omega = np.tile(p1.omega_SS.reshape((1, p1.S)), (p1.T, 1))
BQ1 = 2.5
p1.use_zeta = True
expected1 = np.array([[1.66666667, 7.5], [2.5, 10.0], [0.416666667, 0.0]])
p2 = Specifications()
p2.zeta = np.array([[0.1, 0.3], [0.15, 0.4], [0.05, 0.0]])
p2.S = 3
p2.rho = np.array([[0.0, 0.0, 1.0]])
p2.J = 2
p2.T = 3
p2.lambdas = np.array([0.6, 0.4])
p2.omega_SS = np.array([0.25, 0.25, 0.5])
p2.omega = np.tile(p2.omega_SS.reshape((1, p2.S)), (p2.T, 1))
p2.use_zeta = True
BQ2 = np.array([2.5, 0.8, 3.6])
expected2 = np.array([7.5, 10.0, 0.0])
expected3 = np.array(
    [
        [[1.666666667, 7.5], [2.5, 10.0], [0.416666667, 0.0]],
        [[0.533333333, 2.4], [0.8, 3.2], [0.133333333, 0.0]],
        [[2.4, 10.8], [3.6, 14.4], [0.6, 0.0]],
    ]
)
expected4 = np.array([[7.5, 10.0, 0.0], [2.4, 3.2, 0.0], [10.8, 14.4, 0.0]])
p3 = Specifications()
p3.S = 3
p3.rho = np.array([[0.0, 0.0, 1.0]])
p3.J = 2
p3.T = 3
p3.lambdas = np.array([0.6, 0.4])
p3.omega_SS = np.array([0.25, 0.25, 0.5])
p3.omega = np.tile(p2.omega_SS.reshape((1, p2.S)), (p2.T, 1))
p3.use_zeta = False
BQ3 = np.array([1.1, 0.8])
BQ4 = np.array([[1.1, 0.8], [3.2, 4.6], [2.5, 0.1]])
expected5 = np.array(
    [[1.833333333, 2.0], [1.833333333, 2.0], [1.833333333, 2.0]]
)
expected6 = np.array([2.0, 2.0, 2.0])
expected7 = np.array(
    [
        [[1.833333333, 2.0], [1.833333333, 2.0], [1.833333333, 2.0]],
        [[5.333333333, 11.5], [5.333333333, 11.5], [5.333333333, 11.5]],
        [[4.166666667, 0.25], [4.166666667, 0.25], [4.166666667, 0.25]],
    ]
)
expected8 = np.array([[2.0, 2.0, 2.0], [11.5, 11.5, 11.5], [0.25, 0.25, 0.25]])
test_data = [
    (BQ1, None, p1, "SS", expected1),
    (BQ1, 1, p1, "SS", expected2),
    (BQ2, None, p2, "TPI", expected3),
    (BQ2, 1, p2, "TPI", expected4),
    (BQ3, None, p3, "SS", expected5),
    (BQ3, 1, p3, "SS", expected6),
    (BQ4, None, p3, "TPI", expected7),
    (BQ4, 1, p3, "TPI", expected8),
]


@pytest.mark.parametrize(
    "BQ,j,p,method,expected",
    test_data,
    ids=[
        "SS, use zeta, all j",
        "SS, use zeta, one j",
        "TPI, use zeta, all j",
        "TPI, use zeta, one j",
        "SS, not use zeta, all j",
        "SS, not use zeta, one j",
        "TPI, not use zeta, all j",
        "TPI, not use zeta, one j",
    ],
)
def test_get_bq(BQ, j, p, method, expected):
    # Test the get_bq function
    test_value = household.get_bq(BQ, j, p, method)
    print("Test value = ", test_value)
    assert np.allclose(test_value, expected)

#%%

p1 = Specifications()
p1.eta = np.tile(
    np.array([[0.1, 0.3], [0.15, 0.4], [0.05, 0.0]]).reshape(1, p2.S, p2.J),
    (p2.T, 1, 1),
)
p1.S = 3
p1.J = 2
p1.T = 3
p1.lambdas = np.array([0.6, 0.4])
p1.omega_SS = np.array([0.25, 0.25, 0.5])
p1.omega = np.tile(p1.omega_SS.reshape((1, p1.S)), (p1.T, 1))
TR1 = 2.5
expected1 = np.array([[1.66666667, 7.5], [2.5, 10.0], [0.416666667, 0.0]])
p2 = Specifications()
p2.S = 3
p2.rho = np.array([[0.0, 0.0, 1.0]])
p2.J = 2
p2.T = 3
p2.eta = np.tile(
    np.array([[0.1, 0.3], [0.15, 0.4], [0.05, 0.0]]).reshape(1, p2.S, p2.J),
    (p2.T, 1, 1),
)
p2.lambdas = np.array([0.6, 0.4])
p2.omega_SS = np.array([0.25, 0.25, 0.5])
p2.omega = np.tile(p2.omega_SS.reshape((1, p2.S)), (p2.T, 1))
TR2 = np.array([2.5, 0.8, 3.6])
expected2 = np.array([7.5, 10.0, 0.0])
expected3 = np.array(
    [
        [[1.666666667, 7.5], [2.5, 10.0], [0.416666667, 0.0]],
        [[0.533333333, 2.4], [0.8, 3.2], [0.133333333, 0.0]],
        [[2.4, 10.8], [3.6, 14.4], [0.6, 0.0]],
    ]
)
expected4 = np.array([[7.5, 10.0, 0.0], [2.4, 3.2, 0.0], [10.8, 14.4, 0.0]])
test_data = [
    (TR1, None, p1, "SS", expected1),
    (TR1, 1, p1, "SS", expected2),
    (TR2, None, p2, "TPI", expected3),
    (TR2, 1, p2, "TPI", expected4),
]


@pytest.mark.parametrize(
    "TR,j,p,method,expected",
    test_data,
    ids=["SS, all j", "SS, one j", "TPI, all j", "TPI, one j"],
)
def test_get_tr(TR, j, p, method, expected):
    # Test the get_tr function
    test_value = household.get_tr(TR, j, p, method)
    print("Test value = ", test_value)
    assert np.allclose(test_value, expected)


# %%

def setup_c_from_n_params():
    p = Specifications()
    p.S, p.J, p.T = 5, 2, 10
    p.g_y, p.sigma, p.ltilde = 0.02, 2.0, 1.0
    p.b_ellipse, p.upsilon = 5.0, 2.0
    p.chi_n = np.ones((p.S, p.J))
    p.tau_payroll = np.linspace(0.05, 0.10, p.T)
    p.labor_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.labor_income_tax_noncompliance_rate[:, 1] = 0.10
    e_path = np.tile(
        np.linspace(0.5, 1.5, p.S).reshape(1, p.S, 1), (p.T, 1, p.J)
    )
    e_path[:, :, 1] *= 1.2
    p.e = e_path
    etr_params = np.zeros((p.T, p.S, 12))
    mtrx_params = np.zeros((p.T, p.S, 12))
    mtrx_params[:, :, 10] = 0.25
    return {
        "p": p, "etr_params": etr_params, "mtrx_params": mtrx_params,
        "n": 0.4, "b": 10.0, "p_tilde": 1.0, "r": 0.05, "w": 1.2,
        "factor": 1.0, "z": 1.0,
    }


def get_ss_vector_expected_c(params):
    p, j = params["p"], 0
    S = p.S
    r, w, b, n, z = (np.ones(S) * params[k] for k in ("r", "w", "b", "n", "z"))
    e = np.squeeze(p.e[-1, :, j])
    chi_n = p.chi_n[:, j]
    tau_payroll = p.tau_payroll[-1]
    tax_noncompliance = p.labor_income_tax_noncompliance_rate[-1, j]
    mtrx_params_ss = params["mtrx_params"][-1, :, :]
    mtr_labor = tax.MTR_income(
        r, w, b, n, params["factor"], False, e * z, None, mtrx_params_ss,
        tax_noncompliance, p
    )
    deriv = 1 - tau_payroll - mtr_labor
    mdu_labor = household.marg_ut_labor(n, chi_n, p)
    num = params["p_tilde"] * np.exp(p.g_y * (1 - p.sigma)) * mdu_labor
    den = w * e * z * deriv
    return household.inv_mu_c(num / den, p.sigma)


PARAMS = setup_c_from_n_params()
C_SS_EXPECTED_VEC = get_ss_vector_expected_c(PARAMS)


@pytest.mark.parametrize(
    "method, n, b, r, w, chi_n, mtrx_p, expected_c",
    [
        (
            "SS",
            np.ones(PARAMS["p"].S) * PARAMS["n"],
            np.ones(PARAMS["p"].S) * PARAMS["b"],
            np.ones(PARAMS["p"].S) * PARAMS["r"],
            np.ones(PARAMS["p"].S) * PARAMS["w"],
            PARAMS["p"].chi_n[:, 0],
            PARAMS["mtrx_params"][-1, :, :],
            C_SS_EXPECTED_VEC,
        ),
    ],
    ids=["SS vector"],
)
def test_c_from_n_vec(method, n, b, r, w, chi_n, mtrx_p, expected_c):
    """
    Test of the `c_from_n` function for vectorized cases (SS and TPI).
    This test will only pass with the fixed version of `c_from_n`.
    """
    p = PARAMS["p"]
    p_tilde, factor, z = (PARAMS[k] for k in ("p_tilde", "factor", "z"))

    # For SS, e is calculated internally. For TPI, pass None to trigger internal calc.
    test_c = household.c_from_n(
        n, b, p_tilde, r, w, factor, e=None, z=z, chi_n=chi_n,
        etr_params=PARAMS["etr_params"], mtrx_params=mtrx_p,
        t=None, j=0, p=p, method=method,
    )
    assert np.allclose(test_c, expected_c)

# %%
# Setup for testing b_from_c_EOL
p_b_from_c = Specifications()
p_b_from_c.J = 2
# Set bequest utility weights for two ability types
p_b_from_c.chi_b = np.array([0.5, 0.8])
# Set coefficient of relative risk aversion
p_b_from_c.sigma = 2.0

# Test data format: (c, p_tilde, j, sigma, p, expected_b)
test_data_b_from_c_EOL = [
    # Scenario 1: Scalar inputs
    (1.0, 1.1, 0, 2.0, p_b_from_c, 0.7416198487),
    # Scenario 2: Array c, scalar p_tilde, j=1
    (
        np.array([1.0, 2.0]),
        1.1,
        1,
        2.0,
        p_b_from_c,
        np.array([0.938083152, 1.876166304]),
    ),
    # Scenario 3: Array c and p_tilde
    (
        np.array([1.0, 2.0]),
        np.array([1.1, 1.2]),
        0,
        2.0,
        p_b_from_c,
        np.array([0.7416198487, 1.549193308]),
    ),
    # Scenario 4: Different sigma
    (2.5, 1.0, 1, 3.0, p_b_from_c, 2.5 * (0.8 * 1.0) ** (1 / 3.0)),
]


@pytest.mark.parametrize(
    "c, p_tilde, j, sigma, p, expected_b",
    test_data_b_from_c_EOL,
    ids=["Scalar inputs", "Array c", "Array c and p_tilde", "Different sigma"],
)
def test_b_from_c_EOL(c, p_tilde, j, sigma, p, expected_b):
    """
    Test of the household_stoch.b_from_c_EOL function.
    """
    test_b = household.b_from_c_EOL(c, p_tilde, j, sigma, p)
    assert np.allclose(test_b, expected_b)


# Setup for testing get_cons
p_get_cons = Specifications()
# Set growth rate of technology
p_get_cons.g_y = 0.02

# Test data format: (r, w, p_tilde, b, b_splus1, n, bq, net_tax, e, z, p, expected_cons)
test_data_get_cons = [
    # Case 1: Scalar inputs
    (0.04, 1.5, 1.0, 10.0, 11.0, 0.8, 0.5, 2.0, 1.0, 1.2, p_get_cons, -0.88221474),
    # Case 2: Vector inputs
    (
        np.array([0.04, 0.05]),  # r
        np.array([1.5, 1.6]),  # w
        np.array([1.0, 1.1]),  # p_tilde
        np.array([10.0, 12.0]),  # b
        np.array([11.0, 13.0]),  # b_splus1
        np.array([0.8, 0.7]),  # n
        np.array([0.5, 0.6]),  # bq
        np.array([2.0, 2.5]),  # net_tax
        np.array([1.0, 1.1]),  # e
        np.array([1.2, 0.9]),  # z
        p_get_cons,
        np.array([-0.88221474, -1.3216522]),
    ),
]


@pytest.mark.parametrize(
    "r, w, p_tilde, b, b_splus1, n, bq, net_tax, e, z, p, expected_cons",
    test_data_get_cons,
    ids=["Scalar inputs", "Vector inputs"],
)
def test_get_cons(r, w, p_tilde, b, b_splus1, n, bq, net_tax, e, z, p, expected_cons):
    """
    Test of the household_stoch.get_cons function.
    """
    test_cons = household.get_cons(
        r, w, p_tilde, b, b_splus1, n, bq, net_tax, e, z, p
    )
    assert np.allclose(test_cons, expected_cons)


# %%
def test_ci():
    """
    Test of the get_ci function
    """
    c_s = np.array([2.0, 3.0, 5.0, 7.0]).reshape(4, 1)
    p_i = np.array([1.1, 0.8, 1.0])
    p_tilde = np.array([2.3])
    tau_c = np.array([0.2, 0.3, 0.5])
    alpha_c = np.array([0.5, 0.3, 0.2])
    expected_ci = np.array(
        [
            [1.742424242, 2.613636364, 4.356060606, 6.098484848],
            [1.326923077, 1.990384615, 3.317307692, 4.644230769],
            [0.613333333, 0.92, 1.533333333, 2.146666667],
        ]
    ).reshape(3, 4, 1)

    test_ci = household.get_ci(c_s, p_i, p_tilde, tau_c, alpha_c)

    assert np.allclose(test_ci, expected_ci)

# %%

def setup_c_from_b_splus1_params(taxes=False):
    """
    Set up parameters for testing the c_from_b_splus1 function.
    """
    p = Specifications()
    p.S, p.J, p.T = 5, 1, 5
    p.beta = np.array([0.96])
    p.sigma = 2.0
    p.g_y = 0.02
    p.nz = 2
    p.z_grid = np.array([0.8, 1.2])
    p.Z = np.array([[0.9, 0.1], [0.1, 0.9]])
    p.chi_b = np.array([2.0])
    p.e = np.tile(
        np.linspace(1.0, 1.2, p.S).reshape(1, p.S, 1), (p.T, 1, 1)
    )
    p.retire = p.S + 1 # Everyone works

    # Tax parameters
    p.capital_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    etr_params = np.zeros((p.T, p.S, 12))
    mtry_params = np.zeros((p.T, p.S, 12))
    if taxes:
        # Add a simple MTR on capital income
        mtry_params[:, :, 10] = 0.15
        p.h_wealth = np.ones(p.T) * 0.4
        p.m_wealth = np.ones(p.T) * 1.0
        p.p_wealth = np.ones(p.T) * 0.01
    else:
        p.h_wealth = np.zeros(p.T)
        p.m_wealth = np.ones(p.T)
        p.p_wealth = np.zeros(p.T)

    # Function inputs
    s = 1  # current age is s=1 (second period of life)
    j = 0
    t = 0
    z_index = 0  # current productivity state is low
    r_splus1 = 0.04
    w_splus1 = 1.5
    p_tilde_splus1 = 1.0
    p_tilde_s = 1.0
    b_splus1 = np.array([5.0, 10.0]) # nb=2
    # Next period's policy functions (nb x nz)
    n_splus1_policy = np.array([[0.3, 0.4], [0.3, 0.4]])
    c_splus1_policy = np.array([[2.0, 2.5], [8.0, 9.0]])
    factor = 1.0
    rho_s = 0.1 # mortality rate at age s

    # Manually calculate expected consumption
    beta = p.beta[j]
    bequest_utility = rho_s * household.marg_ut_beq(b_splus1, p.sigma, j, p)
    
    # Calculate expectation of marginal utility
    consumption_utility_matrix = np.zeros((b_splus1.shape[0], p.z_grid.shape[0]))
    for zp_index, zp in enumerate(p.z_grid):
        mtr_capital = tax.MTR_income(
            r_splus1, w_splus1, b_splus1, n_splus1_policy[:, zp_index], factor, True,
            p.e[t, s + 1, j] * zp, etr_params[t, s + 1, :], mtry_params[t, s + 1, :],
            p.capital_income_tax_noncompliance_rate[t, j], p
        )
        mtr_wealth = tax.MTR_wealth(b_splus1, p.h_wealth[t+1], p.m_wealth[t+1], p.p_wealth[t+1])
        deriv = (1 + r_splus1) - (r_splus1 * mtr_capital) - mtr_wealth
        
        mu_c_splus1 = household.marg_ut_cons(c_splus1_policy[:, zp_index], p.sigma)
        consumption_utility_matrix[:, zp_index] = deriv * mu_c_splus1 / p_tilde_splus1
    
    prob_z_splus1 = p.Z[z_index, :]
    E_MU_c = consumption_utility_matrix @ prob_z_splus1

    # Final calculation using the *correct* Euler equation
    growth_term = np.exp(p.g_y * (1 - p.sigma))
    mu_c_rhs = bequest_utility + beta * (1 - rho_s) * growth_term * E_MU_c
    expected_c = household.inv_mu_c(p_tilde_s * mu_c_rhs, p.sigma)

    # Gather args for function call
    args = (
        r_splus1, w_splus1, p_tilde_splus1, p_tilde_s, b_splus1,
        n_splus1_policy, c_splus1_policy, factor, rho_s,
        etr_params[t+1, s + 1, :], mtry_params[t+1, s + 1, :], j, t+1,
        p.e[t, s + 1, j], z_index, p, "TPI"
    )

    return args, expected_c

@pytest.mark.parametrize(
    "setup_params",
    [
        {"taxes": False},
        {"taxes": True}
    ],
    ids=["No Taxes", "With Capital/Wealth Taxes"]
)
def test_c_from_b_splus1(setup_params):
    """
    Test of the household_stoch.c_from_b_splus1 function.
    """
    args, expected_c = setup_c_from_b_splus1_params(**setup_params)
    
    test_c = household.c_from_b_splus1(*args)
    
    assert np.allclose(test_c, expected_c)

# %% Tests for FOC_labor and get_y


def test_get_y_stoch():
    """
    Test of the household_stoch.get_y() function.
    """
    # Setup parameters
    p = Specifications()
    p.S = 3
    p.J = 1
    p.T = 1
    # Set a 3D e-path as the function expects
    p.e = np.array([[[1.0], [1.5], [0.2]]])

    # Test case variables
    r_p = np.array([0.05, 0.04, 0.09])
    w = np.array([1.2, 0.8, 2.5])
    b_s = np.array([0.5, 0.99, 9])
    n = np.array([0.8, 3.2, 0.2])
    # Case 1: Scalar productivity shock
    z1 = 1.1
    # Case 2: Vector of productivity shocks
    z2 = np.array([0.9, 1.0, 1.2])

    # Manually calculate expected income
    e_ss = np.squeeze(p.e[-1, :, :])
    expected_y1 = r_p * b_s + w * e_ss * z1 * n
    expected_y2 = r_p * b_s + w * e_ss * z2 * n

    # Run tests
    test_y1 = household.get_y(r_p, w, b_s, n, z1, p, "SS")
    test_y2 = household.get_y(r_p, w, b_s, n, z2, p, "SS")

    assert np.allclose(test_y1, expected_y1)
    assert np.allclose(test_y2, expected_y2)


# Setup for FOC_labor test
# Define variables for test of SS version
p_foc = Specifications()
p_foc.sigma = 1.5
p_foc.g_y = 0.04
p_foc.b_ellipse = 0.527
p_foc.upsilon = 1.45
p_foc.ltilde = 1.2
p_foc.J = 1
p_foc.S = 3
p_foc.T = 3
p_foc.chi_n = np.array([0.75, 0.8, 0.9]).reshape(3, 1)
p_foc.e = np.array([1.0, 0.9, 1.4]).reshape(3, 1)
# Make e 3D for TPI cases if needed, but for SS it's okay
p_foc.e = np.tile(p_foc.e.reshape(1, p_foc.S, p_foc.J), (p_foc.T, 1, 1))

p_foc.labor_income_tax_noncompliance_rate = np.zeros((p_foc.T + p_foc.S, p_foc.J))
p_foc.tau_payroll = np.array([0.15])
# Set up simple tax functions
etr_params = np.zeros((p_foc.S, 12))
mtrx_params = np.zeros((p_foc.S, 12))
mtrx_params[:, 10] = 0.22  # Simple 22% MTR on labor income
# Make tax params 3D for TPI cases
etr_params_3d = np.tile(etr_params.reshape(1, p_foc.S, 12), (p_foc.T, 1, 1))
mtrx_params_3d = np.tile(mtrx_params.reshape(1, p_foc.S, 12), (p_foc.T, 1, 1))


# Variables for the test
r = 0.05
w = 1.2
b = np.array([0.0, 0.8, 0.5])
n = np.array([0.9, 0.8, 0.5])
b_splus1 = np.array([0.8, 0.5, 0.1])
bq = np.array([0.1, 0.1, 0.1])
factor = 1.0
tr = 0.0
ubi = 0.0
theta = np.array([0.0])
p_tilde = 1.0
j = 0
t = None  # for SS
method = "SS"
z_scalar = 1.1
z_vector = np.array([0.9, 1.0, 1.2])

# To correctly test, we need a consistent consumption value
e_ss = np.squeeze(p_foc.e[-1, :, j])
net_tax_scalar_z = tax.net_taxes(
    r, w, b, n, bq, factor, tr, ubi, theta, t, j, False, method, e_ss * z_scalar, etr_params, p_foc
)
c_scalar_z = household.get_cons(
    r, w, p_tilde, b, b_splus1, n, bq, net_tax_scalar_z, e_ss, z_scalar, p_foc
)

net_tax_vector_z = tax.net_taxes(
    r, w, b, n, bq, factor, tr, ubi, theta, t, j, False, method, e_ss * z_vector, etr_params, p_foc
)
c_vector_z = household.get_cons(
    r, w, p_tilde, b, b_splus1, n, bq, net_tax_vector_z, e_ss, z_vector, p_foc
)

# Manually calculate FOC error for the scalar case
mtrx_s = tax.MTR_income(
    r, w, b, n, factor, False, e_ss * z_scalar, etr_params, mtrx_params, 0.0, p_foc
)
deriv_s = 1 - p_foc.tau_payroll[-1] - mtrx_s
mu_c_s = household.marg_ut_cons(c_scalar_z, p_foc.sigma)
mdu_n_s = household.marg_ut_labor(n, np.squeeze(p_foc.chi_n), p_foc)
expected_ss_scalar_z = mu_c_s * (1 / p_tilde) * w * deriv_s * e_ss * z_scalar - mdu_n_s

# Manually calculate FOC error for the vector case
mtrx_v = tax.MTR_income(
    r, w, b, n, factor, False, e_ss * z_vector, etr_params, mtrx_params, 0.0, p_foc
)
deriv_v = 1 - p_foc.tau_payroll[-1] - mtrx_v
mu_c_v = household.marg_ut_cons(c_vector_z, p_foc.sigma)
mdu_n_v = household.marg_ut_labor(n, np.squeeze(p_foc.chi_n), p_foc)
expected_ss_vector_z = mu_c_v * (1 / p_tilde) * w * deriv_v * e_ss * z_vector - mdu_n_v


@pytest.mark.parametrize(
    "c, z, e, expected",
    [
        (c_scalar_z, z_scalar, e_ss, expected_ss_scalar_z),
        (c_vector_z, z_vector, e_ss, expected_ss_vector_z),
    ],
    ids=["SS with scalar z", "SS with vector z"],
)
def test_FOC_labor_stoch(c, z, e, expected):
    """
    Test of the household_stoch.FOC_labor() function.
    """
    test_error = household.FOC_labor(
        r,
        w,
        p_tilde,
        b,
        c,
        n,
        factor,
        e,
        z,
        np.squeeze(p_foc.chi_n),
        etr_params,
        mtrx_params,
        t,
        j,
        p_foc,
        method,
    )

    assert np.allclose(test_error, expected)

# %%

bssmat0 = np.array([[0.1, 0.2], [0.3, 0.4]])
nssmat0 = np.array([[0.1, 0.2], [0.3, 0.4]])
cssmat0 = np.array([[0.1, 0.2], [0.3, 0.4]])

bssmat1 = np.array([[-0.1, -0.2], [-0.3, -0.4]])
nssmat1 = np.array([[-0.1, -0.2], [3.3, 4.4]])
cssmat1 = np.array([[-0.1, -0.2], [-0.3, -0.4]])
test_data = [
    (bssmat0, nssmat0, cssmat0, 1.0),
    (bssmat1, nssmat1, cssmat1, 1.0),
]


@pytest.mark.parametrize(
    "bssmat,nssmat,cssmat,ltilde", test_data, ids=["passing", "failing"]
)
def test_constraint_checker_SS(bssmat, nssmat, cssmat, ltilde):
    household.constraint_checker_SS(bssmat, nssmat, cssmat, ltilde)
    assert True


@pytest.mark.parametrize(
    "bssmat,nssmat,cssmat,ltilde", test_data, ids=["passing", "failing"]
)
def test_constraint_checker_TPI(bssmat, nssmat, cssmat, ltilde):
    household.constraint_checker_TPI(bssmat, nssmat, cssmat, 10, ltilde)
    assert True

# %% Test for BC_residual

# Setup parameters and variables for a series of test cases
p_bc = Specifications()
p_bc.g_y = 0.02  # Set a non-zero growth rate
savings_detrend_factor = np.exp(p_bc.g_y)

# Test Case 1: Scalar inputs, zero residual
r1, w1, p_tilde1 = 0.04, 1.5, 1.0
b1, n1, b_splus1_1 = 10.0, 0.8, 11.0
bq1, net_tax1, e1, z1 = 0.5, 2.0, 1.0, 1.2
# Manually calculate the consumption level that solves the budget constraint
resources1 = (1 + r1) * b1 + w1 * e1 * z1 * n1 + bq1 - net_tax1
c1_zero_resid = (resources1 - b_splus1_1 * savings_detrend_factor) / p_tilde1
args1 = (
    c1_zero_resid, n1, b1, b_splus1_1, r1, w1, p_tilde1, e1, z1, bq1,
    net_tax1, p_bc
)

# Test Case 2: Scalar inputs, positive residual (underspending)
c2_pos_resid = c1_zero_resid - 1.0  # Consume less than budget allows
args2 = (
    c2_pos_resid, n1, b1, b_splus1_1, r1, w1, p_tilde1, e1, z1, bq1,
    net_tax1, p_bc
)
# The residual should be the unspent amount: p_tilde * 1.0
expected2 = p_tilde1 * 1.0

# Test Case 3: Vector inputs, zero residual
r3 = np.array([0.04, 0.05])
w3, p_tilde3 = np.array([1.5, 1.6]), np.array([1.0, 1.1])
b3, n3 = np.array([10.0, 12.0]), np.array([0.8, 0.7])
b_splus1_3 = np.array([11.0, 13.0])
bq3, net_tax3 = np.array([0.5, 0.6]), np.array([2.0, 2.5])
e3, z3 = np.array([1.0, 1.1]), np.array([1.2, 0.9])
# Calculate the c vector that solves the budget constraint
resources3 = (1 + r3) * b3 + w3 * e3 * z3 * n3 + bq3 - net_tax3
c3_zero_resid = (resources3 - b_splus1_3 * savings_detrend_factor) / p_tilde3
args3 = (
    c3_zero_resid, n3, b3, b_splus1_3, r3, w3, p_tilde3, e3, z3, bq3,
    net_tax3, p_bc
)

# Test Case 4: Vector inputs, non-zero residual (overspending)
c4_neg_resid = c3_zero_resid + np.array([0.5, 2.0])  # Consume more than budget allows
args4 = (
    c4_neg_resid, n3, b3, b_splus1_3, r3, w3, p_tilde3, e3, z3, bq3,
    net_tax3, p_bc
)
# The residual should be the negative of the overspent amount
expected4 = -p_tilde3 * np.array([0.5, 2.0])


@pytest.mark.parametrize(
    "args, expected_residual",
    [
        (args1, 0.0),
        (args2, expected2),
        (args3, np.array([0.0, 0.0])),
        (args4, expected4),
    ],
    ids=[
        "Scalar, zero residual",
        "Scalar, positive residual",
        "Vector, zero residual",
        "Vector, non-zero residual",
    ],
)
def test_BC_residual(args, expected_residual):
    """
    Test of the household_stoch.BC_residual function.
    """
    test_residual = household.BC_residual(*args)
    assert np.allclose(test_residual, expected_residual)

# %% Test for EOL_system


def setup_EOL_system_params(vector=False):
    """
    Set up parameters for testing the EOL_system function.
    This creates a consistent set of parameters and variables, then
    manually calculates the expected residual from the budget constraint
    at the end of life, which is what EOL_system is designed to return.
    """
    p = Specifications()
    p.sigma = 2.0
    p.g_y = 0.02
    p.b_ellipse = 5.0
    p.upsilon = 2.0
    p.ltilde = 1.0
    p.J = 1
    p.S = 3
    p.T = 3
    # p.retire = p.S 
    p.retire = np.array([p.S]) #Changed to array
    # Use s-specific chi_n and 3D e, as EOL is age-specific
    p.chi_n = np.array([0.8, 0.85, 0.9]).reshape(3, 1)
    p.e = np.array([1.0, 1.1, 1.2]).reshape(3, 1)
    p.e = np.tile(p.e.reshape(1, p.S, p.J), (p.T, 1, 1))
    # Use j-specific chi_b
    p.chi_b = np.array([2.5])

    # Simplified tax system for clarity
    p.labor_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.tau_payroll = np.array([0.1, 0.1, 0.1])
    # The functions called by EOL_system expect tax params for a specific age.
    etr_params_s = np.zeros(12)
    mtrx_params_s = np.zeros(12)
    mtrx_params_s[10] = 0.20  # Flat 20% MTR on labor

    # --- Function inputs ---
    s_EOL = p.S - 1  # End of life period index (s=2)
    r = 0.04
    w = np.array([1.5])
    p_tilde = 1.0
    tr = 0.1
    ubi = 0.0
    bq = 0.2
    theta = np.array([0.0])
    factor = 1.0
    j = 0
    method = "SS"
    # For SS method, t is not used for indexing. Can be None or 0 if not used for indexing.
    t = None

    # Get values for the specific EOL age
    e_s = p.e[-1, s_EOL, j]
    chi_n_s = p.chi_n[s_EOL, j]

    if not vector:
        # n = np.array([0.5])
        n = 0.5
        b = 8.0
        z = 1.1
    else:
        n = np.array([0.4, 0.6])
        b = np.array([7.0, 9.0])
        z = np.array([0.9, 1.2])

    n = np.atleast_1d(n)  # Ensure n is always an array for pension
    # Manually calculate the expected residual by replicating EOL_system logic
    # 1. Consumption from labor FOC
    c = household.c_from_n(
        n, b, p_tilde, r, w, factor, e_s, z, chi_n_s, etr_params_s,
        mtrx_params_s, t=t, j=j, p=p, method=method
    )

    # 2. Bequests from consumption
    b_splus1 = household.b_from_c_EOL(c, p_tilde, j, p.sigma, p)

    # 3. Net taxes
    net_tax = tax.net_taxes(
        r, w, b, n, bq, factor, tr, ubi, theta, t, j, False,
        method, e_s * z, etr_params_s, p
    )

    # 4. Budget constraint residual
    expected_residual = household.BC_residual(
        c, n, b, b_splus1, r, w, p_tilde, e_s, z, bq, net_tax, p
    )

    # Gather args for the function call to EOL_system
    args = (
        n, b, p_tilde, r, w, tr, ubi, bq, theta, factor, e_s, z,
        chi_n_s, etr_params_s, mtrx_params_s, t, j, p, method
    )

    return args, expected_residual


@pytest.mark.parametrize(
    "vector_case", [False, True], ids=["Scalar case", "Vector case"]
)
def test_EOL_system(vector_case):
    """
    Test of the household_stoch.EOL_system function.
    """
    args, expected_residual = setup_EOL_system_params(vector=vector_case)

    test_residual = household.EOL_system(*args)

    assert np.allclose(test_residual, expected_residual)

# %% Test for HH_system

def setup_HH_system_params(scalar_case=True):
    """
    Set up parameters for testing the HH_system function.
    This creates a consistent set of parameters and variables, then
    manually calculates the expected residuals from the budget constraint
    and the labor FOC, which is what HH_system is designed to return.
    """
    p = Specifications()
    p.sigma, p.g_y, p.ltilde = 1.5, 0.02, 1.2
    p.b_ellipse, p.upsilon = 0.527, 1.45
    p.J, p.S, p.T = 1, 3, 3
    p.chi_n = np.array([0.75, 0.8, 0.9]).reshape(3, 1)
    p.e = np.array([1.0, 0.9, 1.4]).reshape(3, 1)
    p.e = np.tile(p.e.reshape(1, p.S, p.J), (p.T, 1, 1))

    # Simplified tax system
    p.labor_income_tax_noncompliance_rate = np.zeros((p.T, p.J))
    p.tau_payroll = np.array([0.15])
    etr_params_s = np.zeros(12)
    mtrx_params_s = np.zeros(12)
    mtrx_params_s[10] = 0.22  # Flat 22% MTR on labor

    # Function inputs
    s = 1  # A period before the end of life
    r, w, p_tilde = 0.05, 1.2, 1.0
    tr, ubi, bq = 0.0, 0.0, 0.1
    theta = np.array([0.1])
    factor = 1.0
    j, t, method = 0, None, "SS"
    e_s = p.e[-1, s, j]
    chi_n_s = p.chi_n[s, j]

    if scalar_case:
        x = np.array([8.0, 0.7])  # guess for [b, n]
        c = 7.5
        b_splus1 = 8.2
        z = 1.1
    else: # vector case for x, c, b_splus1, and z
        x = np.array([np.array([8.0, 9.0]), np.array([0.7, 0.6])])
        c = np.array([7.5, 8.5])
        b_splus1 = np.array([8.2, 9.3])
        z = np.array([1.1, 0.9])

    b, n = x[0], x[1]

    # need n as array for pension calculations
    n = np.atleast_1d(n)

    # Manually calculate expected residuals 
    net_tax = tax.net_taxes(
        r, w, b, n, bq, factor, tr, ubi, theta, t, j, False,
        method, e_s * z, etr_params_s, p
    )
    BC_error = household.BC_residual(
        c, n, b, b_splus1, r, w, p_tilde, e_s, z, bq, net_tax, p
    )
    FOC_error = household.FOC_labor(
        r, w, p_tilde, b, c, n, factor, e_s, z, chi_n_s, etr_params_s,
        mtrx_params_s, t, j, p, method
    )

    expected_HH_error = np.array([BC_error, FOC_error])

    # Gather args for the function call
    args = (
        x, c, b_splus1, r, w, p_tilde, factor, tr, ubi, bq, theta, e_s, z,
        chi_n_s, etr_params_s, mtrx_params_s, j, t, p, method
    )
    
    # THE FIX: Squeeze the expected error for the scalar case to match the
    # shape of the function's output. The vector case is returned as is.
    if scalar_case:
        return args, np.squeeze(expected_HH_error)
    else:
        return args, expected_HH_error


@pytest.mark.parametrize(
    "scalar_case", [True, False], ids=["Scalar inputs", "Vector inputs"]
)
def test_HH_system(scalar_case):
    """
    Test of the household_stoch.HH_system function.
    This function computes the residuals for the budget constraint and the
    labor supply FOC for a given guess of savings (b) and labor (n).
    """
    args, expected_error = setup_HH_system_params(scalar_case=scalar_case)

    # The HH_system function expects x to be a 1D array/list.
    # For the vector case, we cannot pass a 2D array, so we must loop.
    if not scalar_case:
        # Unpack args and test each element of the vector case individually
        (x, c, b_splus1, r, w, p_tilde, factor, tr, ubi, bq, theta, e_s, z,
        chi_n_s, etr_params_s, mtrx_params_s, j, t, p, method) = args

        for i in range(len(x[0])):
            x_i = np.array([x[0][i], x[1][i]])
            args_i = (
                x_i, c[i], b_splus1[i], r, w, p_tilde, factor, tr, ubi,
                bq, theta, e_s, z[i], chi_n_s, etr_params_s,
                mtrx_params_s, j, t, p, method
            )
            test_error_i = household.HH_system(*args_i)
            expected_error_i = expected_error[:, i]
            assert np.allclose(test_error_i, expected_error_i)
    else: # Scalar case
        test_error = household.HH_system(*args)
        assert np.allclose(test_error, expected_error)

# %% Test for solve_HH

# Create a Specifications object for the test
p_solve_hh = Specifications()
p_solve_hh.S = 4  # Life periods
p_solve_hh.J = 1  # Ability types
p_solve_hh.T = 4  # Time periods for TPI
p_solve_hh.beta = np.array([0.96])
p_solve_hh.sigma = 2.0
p_solve_hh.g_y = 0.01
p_solve_hh.nz = 2  # Number of productivity states
p_solve_hh.z_grid = np.array([0.8, 1.2])
p_solve_hh.Z = np.array([[0.9, 0.1], [0.1, 0.9]])  # Markov transition matrix
p_solve_hh.chi_b = np.array([2.0])
p_solve_hh.ltilde = 1.0
p_solve_hh.b_ellipse = 5.0
p_solve_hh.upsilon = 2.0
p_solve_hh.retire = np.array([p_solve_hh.S + 1]) # need as array?

# Set 3D and 2D arrays for parameters as expected by functions
p_solve_hh.e = np.tile(
    np.linspace(1.0, 1.2, p_solve_hh.S).reshape(1, p_solve_hh.S, 1),
    (p_solve_hh.T, 1, 1)
)
p_solve_hh.chi_n = np.tile(
    np.array([0.5, 0.6, 0.7, 0.8]).reshape(1, p_solve_hh.S, 1),
    (p_solve_hh.T, 1, 1)
)
p_solve_hh.rho = np.array([0.1, 0.1, 0.1, 1.0])  # Age-specific mortality rates

# Tax parameters
p_solve_hh.labor_income_tax_noncompliance_rate = np.zeros((p_solve_hh.T, p_solve_hh.J))
p_solve_hh.capital_income_tax_noncompliance_rate = np.zeros((p_solve_hh.T, p_solve_hh.J))
etr_params_hh = np.zeros((p_solve_hh.T, p_solve_hh.S, 12))
mtrx_params_hh = np.zeros((p_solve_hh.T, p_solve_hh.S, 12))
mtry_params_hh = np.zeros((p_solve_hh.T, p_solve_hh.S, 12))
mtrx_params_hh[:, :, 10] = 0.20  # 20% MTR on labor
mtry_params_hh[:, :, 10] = 0.10  # 10% MTR on capital
p_solve_hh.h_wealth = np.zeros(p_solve_hh.T)
p_solve_hh.m_wealth = np.ones(p_solve_hh.T)
p_solve_hh.p_wealth = np.zeros(p_solve_hh.T)
p_solve_hh.tau_payroll = np.ones(p_solve_hh.T) * 0.05

# Other inputs for solve_HH for a steady-state (SS) case
r_hh = 0.04
w_hh = 1.2
p_tilde_hh = 1.0
factor_hh = 1.0
tr_hh = 0.0
bq_hh = 0.0
ubi_hh = 0.0
b_grid_hh = np.linspace(0.001, 20, 10)
theta_hh = np.array([0.0])
j_hh = 0
t_hh = 0  # Start time for TPI path

# For SS, functions expect S-length vectors for time-varying params
ss_r = np.ones(p_solve_hh.S) * r_hh
ss_w = np.ones(p_solve_hh.S) * w_hh
ss_p_tilde = np.ones(p_solve_hh.S) * p_tilde_hh
ss_tr = np.ones(p_solve_hh.S) * tr_hh
ss_bq = np.ones(p_solve_hh.S) * bq_hh
# The 't' parameter in solve_HH is used for indexing TPI paths. For SS, it can be a vector of zeros.
ss_t = np.zeros(p_solve_hh.S, dtype=int)
ss_e = p_solve_hh.e[-1, :, j_hh]
ss_chi_n = p_solve_hh.chi_n[-1, :, j_hh]
ss_etr_params = etr_params_hh[-1, :, :]
ss_mtrx_params = mtrx_params_hh[-1, :, :]
ss_mtry_params = mtry_params_hh[-1, :, :]

# This dictionary holds all arguments for the call to solve_HH
# This is not a fixture, just a dictionary for organizing args
solve_hh_args = {
    "r": ss_r, "w": ss_w, "p_tilde": ss_p_tilde, "factor": factor_hh,
    "tr": ss_tr, "bq": ss_bq, "ubi": ubi_hh, "b_grid": b_grid_hh,
    "sigma": p_solve_hh.sigma, "theta": theta_hh, "chi_n": ss_chi_n,
    "rho": p_solve_hh.rho, "e": ss_e, "etr_params": ss_etr_params,
    "mtrx_params": ss_mtrx_params, "mtry_params": ss_mtry_params,
    "j": j_hh, "t": ss_t, "p": p_solve_hh, "method": "SS"
}


def test_solve_HH():
    """
    Test of the household_stoch.solve_HH function.
    This test runs the solver and checks for plausible properties of the
    resulting policy functions, such as shape, monotonicity, and that
    the choices satisfy the household's first-order conditions.
    """
    # Use the pre-defined arguments from the module scope
    args = solve_hh_args
    p = args["p"]
    b_grid = args["b_grid"]
    nb = len(b_grid)

    # Run the solver
    b_policy, c_policy, n_policy = household.solve_HH(**args)

    # 1. Test output shapes and types
    assert isinstance(b_policy, np.ndarray)
    assert isinstance(c_policy, np.ndarray)
    assert isinstance(n_policy, np.ndarray)
    expected_shape = (p.S, nb, p.nz)
    assert b_policy.shape == expected_shape
    assert c_policy.shape == expected_shape
    assert n_policy.shape == expected_shape

    # 2. Test for constraint satisfaction
    assert np.all(c_policy >= 0)
    assert np.all(b_policy >= 0)  # Assumes borrowing constraint b >= 0
    assert np.all(n_policy >= 0)
    assert np.all(n_policy <= p.ltilde)

    # 3. Test for monotonicity
    # Savings and consumption should be increasing in assets for each age and shock
    for s in range(p.S):
        for z in range(p.nz):
            # Differences between adjacent elements in the policy should be non-negative
            assert np.all(np.diff(c_policy[s, :, z]) >= 0)
            assert np.all(np.diff(b_policy[s, :, z]) >= 0)

    # 4. Test consistency by checking FOC and BC for a sample point
    # Pick a point in the middle of the grid, away from the boundaries
    s_test = p.S - 2  # Not the last period
    b_idx_test = nb // 2
    z_idx_test = 0
    b_s = b_grid[b_idx_test]
    n_s = n_policy[s_test, b_idx_test, z_idx_test]
    c_s = c_policy[s_test, b_idx_test, z_idx_test]
    b_splus1 = b_policy[s_test, b_idx_test, z_idx_test]
    z_s = p.z_grid[z_idx_test]

    # Check the budget constraint for this point
    net_tax = tax.net_taxes(
        args["r"][s_test], args["w"][s_test], b_s, n_s, args["bq"][s_test],
        args["factor"], args["tr"][s_test], args["ubi"], args["theta"],
        args["t"][s_test], args["j"], False, "SS", args["e"][s_test] * z_s,
        args["etr_params"][s_test, :], args["p"]
    )
    bc_resid = household.BC_residual(
        c_s, n_s, b_s, b_splus1, args["r"][s_test], args["w"][s_test],
        args["p_tilde"][s_test], args["e"][s_test], z_s, args["bq"][s_test],
        net_tax, p
    )

    # Check the labor FOC for this point
    foc_lab_resid = household.FOC_labor(
        args["r"][s_test], args["w"][s_test], args["p_tilde"][s_test],
        b_s, c_s, n_s, args["factor"], args["e"][s_test], z_s,
        args["chi_n"][s_test], args["etr_params"][s_test, :],
        args["mtrx_params"][s_test, :], args["t"][s_test], args["j"], p, "SS"
    )

    # The residuals should be very close to zero
    assert np.isclose(bc_resid, 0, atol=1e-5)
    assert np.isclose(foc_lab_resid, 0, atol=1e-5)