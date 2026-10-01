"""
------------------------------------------------------------------------
Household functions with idiosyncratic income shocks.

Functions that are unchanged from OG-Core are imported from
``ogcore.household`` rather than copied, so they track OG-Core master.
They are re-exported here so ``household_stoch.<name>`` keeps working.
------------------------------------------------------------------------
"""

# Packages
import numpy as np
import scipy.interpolate as itp
import scipy.optimize as opt
from ogcore import tax
from ogcore.household import (  # noqa: F401 (re-exported)
    constraint_checker_SS,
    constraint_checker_TPI,
    get_bq,
    get_ci,
    get_tr,
    marg_ut_cons,
    marg_ut_labor,
)

# Names this module takes unchanged from ogcore.household
OGCORE_HOUSEHOLD_FUNCTIONS = (
    "constraint_checker_SS",
    "constraint_checker_TPI",
    "get_bq",
    "get_ci",
    "get_tr",
    "marg_ut_cons",
    "marg_ut_labor",
)

"""
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
"""


def set_shock_process(p, z_grid, z_trans):
    r"""
    Attach a discrete Markov process for idiosyncratic productivity
    shocks :math:`z` to a Specifications object.

    Sets ``p.nz``, ``p.z_grid`` and ``p.z_trans``. The transition
    matrix is stored as ``p.z_trans`` because ``p.Z`` is total factor
    productivity in OG-Core. These attributes become regular parameters
    in ``default_parameters.json`` when the solver is ported (Phase 3).

    Args:
        p (OG-Core Specifications object): model parameters, modified
            in place
        z_grid (array_like): productivity levels, shape (nz,)
        z_trans (array_like): Markov transition matrix, shape (nz, nz),
            with ``z_trans[i, k]`` = Pr(z' = z_grid[k] | z = z_grid[i])

    Returns:
        p (OG-Core Specifications object): the same object, for chaining

    Raises:
        ValueError: if shapes do not match or rows are not probabilities
    """
    z_grid = np.atleast_1d(np.asarray(z_grid, dtype=float))
    z_trans = np.atleast_2d(np.asarray(z_trans, dtype=float))
    nz = z_grid.shape[0]
    if z_grid.ndim != 1:
        raise ValueError(f"z_grid must be 1-D, got shape {z_grid.shape}")
    if z_trans.shape != (nz, nz):
        raise ValueError(
            f"z_trans must have shape ({nz}, {nz}), got {z_trans.shape}"
        )
    if (z_trans < 0).any() or not np.allclose(z_trans.sum(axis=1), 1.0):
        raise ValueError("Rows of z_trans must be probability vectors")
    p.nz = nz
    p.z_grid = z_grid
    p.z_trans = z_trans
    return p


def marg_ut_beq(b, sigma, j, p):
    r"""
    Compute the marginal utility of savings.

    .. math::
        MU_{b} = \chi^b_{j}b_{j,s,t}^{-\sigma}

    Args:
        b (array_like): household savings
        sigma (float): coefficient of relative risk aversion
        j (int): index of lifetime income group
        p (OG-Core Specifications object): model parameters

    Returns:
        output (array_like): marginal utility of savings

    """
    if np.ndim(b) == 0:
        b = np.array([b])
    epsilon = 0.001
    bvec_cnstr = b < epsilon
    MU_b = np.zeros(b.shape)
    MU_b[~bvec_cnstr] = p.chi_b[j] * b[~bvec_cnstr] ** (-sigma)

    # This is the linear approximation of b**(-sigma)
    b2 = -sigma * (epsilon ** (-sigma - 1))
    b1 = (epsilon ** (-sigma)) - b2 * epsilon
    MU_b[bvec_cnstr] = p.chi_b[j] * (b2 * b[bvec_cnstr] + b1)

    output = MU_b
    output = np.squeeze(output)
    return output


def inv_mu_c(value, sigma):
    r"""
    Compute the inverse of the marginal utility of consumption, with left
    stitching.

    .. math::
       c = val^{-1/\sigma}

    Args:
        value (array_like): marginal utility of consumption
        sigma (scalar): coefficient of relative risk aversion

    Returns:
        output (array_like): household consumption

    """
    if np.ndim(value) == 0:
        value = np.array([value])

    # Include stitching
    epsilon = 0.001
    val_cnstr = value < epsilon
    output = np.zeros(value.shape)
    output[~val_cnstr] = value[~val_cnstr] ** (-1 / sigma)

    # The approximation is L(v) = f(epsilon) + f'(epsilon) * (v - epsilon)
    # where f(v) = v**(-1/sigma)
    slope = (-1 / sigma) * (epsilon ** (-1 / sigma - 1))
    b1 = (epsilon ** (-1 / sigma)) - slope * epsilon  # Intercept
    output[val_cnstr] = slope * value[val_cnstr] + b1

    output = np.squeeze(output)
    return output


def c_from_n(
    n,
    b,
    p_tilde,
    r,
    w,
    factor,
    e,
    z,
    chi_n,
    etr_params,
    mtrx_params,
    t,
    j,
    p,
    method,
):
    r"""
    Calculate household consumption from labor supply Euler equation for
    group j.

    .. math::
        c_{j,s,t} = \left[ \frac{p_t e^{g_y(1-\sigma)}\chi_s^n h'(n_{j,s,t})}{
        w_t e_{j, s}z_{j, s}(1- \tau^{mtrx}_{s,t})} \right]^{-1/\sigma}

    Args:
        n (array_like): household labor supply
        b (array_like): household savings
        p_tilde (array_like): composite good price
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        factor (scalar): scaling factor converting model units to dollars
        e (array_like): effective labor units (deterministic)
        z (array_like): productivity (stochastic)
        chi_n (array_like): utility weight on the disutility of labor
        etr_params (list): parameters of the effective tax rate
            functions
        mtrx_params (list): parameters of the marginal tax rate
            on labor income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        c (array_like): consumption implied by labor choice
    """
    # simplified logic for tax rates
    if method == "SS":
        tau_payroll = p.tau_payroll[-1]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[-1, j]
        income_tax_filer = p.income_tax_filer[-1, j]
        if e is None:
            e = np.squeeze(p.e[-1, :, j])
    elif method == "TPI":
        # When t is a scalar, we are solving for a particular period
        tau_payroll = p.tau_payroll[t]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[t, j]
        income_tax_filer = p.income_tax_filer[t, j]
        if e is None:
            # Simplified for TPI, assuming e is passed correctly for age s
            # The original logic was overly complex for this function's scope
            e = p.e[t, :, j]
    else:  # Fallback for other methods if needed
        tau_payroll = p.tau_payroll[0]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[0, j]
        income_tax_filer = p.income_tax_filer[0, j]

    deriv = (
        1
        - tau_payroll
        - tax.MTR_income(
            r,
            w,
            b,
            n,
            factor,
            False,
            e * z,
            etr_params,
            mtrx_params,
            tax_noncompliance,
            income_tax_filer,
            p,
        )
    )
    numerator = p_tilde * marg_ut_labor(n, chi_n, p)
    denominator = w * e * z * deriv
    c = inv_mu_c(numerator / denominator, p.sigma)

    return c


def b_from_c_EOL(c, p_tilde, j, sigma, p):
    r"""
    Calculate household bequests at the end of life from the savings Euler
    equation.

    .. math::
        b_{j, E+S+1, t+1} =
        [\chi_j^b \tilde p_t]^{\frac{1}{\sigma}} * c_{j, E+S, t}

    Args:
        c (array_like): household consumption
        p_tilde (array_like): composite good price
        j (int): index of ability type
        sigma (scalar): coefficient of relative risk aversion
        p (OG-Core Specifications object): model parameters

    Returns:
        b (array_like): household savings at the end of life
    """
    b = (c * (p.chi_b[j] * p_tilde) ** (1 / sigma)) / np.exp(p.g_y)
    return b


def get_cons(r, w, p_tilde, b, b_splus1, n, bq, net_tax, e, z, p):
    r"""
    Calculate household consumption.

    .. math::
        c_{j,s,t} =  \frac{(1 + r_{t})b_{j,s,t} + w_t e_{j,s} n_{j,s,t}
        + bq_{j,s,t} + tr_{j,s,t} - T_{j,s,t} -
        e^{g_y}b_{j,s+1,t+1}}{1 - \tau^{c}_{s,t}}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): the ratio of real GDP to nominal GDP
        b (Numpy array): household savings
        b_splus1 (Numpy array): household savings one period ahead
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        net_tax (Numpy array): household net taxes paid
        e (Numpy array): effective labor units
        z (array_like): labor productivity
        p (OG-Core Specifications object): model parameters

    Returns:
        cons (Numpy array): household consumption

    """
    cons = (
        (1 + r) * b + w * e * z * n + bq - b_splus1 * np.exp(p.g_y) - net_tax
    ) / p_tilde  # TODO: add consumption taxes, remittances, pension income
    return cons


def c_from_b_splus1(
    r_splus1,
    w_splus1,
    p_tilde_splus1,
    p_tilde,
    b_splus1,
    n_splus1_policy,
    c_splus1_policy,
    factor,
    rho,
    etr_params,
    mtry_params,
    j,
    t,
    e_splus1,
    z_index,
    p,
    method,
):
    r"""
    Calculate household consumption in period s from assets at period s+1 using
    the savings Euler equation.

    .. math::
        c_{j,s,t} = (\tilde{p}_t)^{-\frac{1}{\sigma}} e^{g_y}
        \biggl[\chi^b_j\rho_s(b_{j,s+1,t+1})^{-\sigma} +
        \beta_j\bigl(1 - \rho_s\bigr)\Bigl(\frac{1 + r_{t+1}
        \bigl[1 - \tau^{mtry}_{s+1,t+1}\bigr]}{\tilde{p}_{t+1}}\Bigr)
        \mathbb{E}[(c_{j,s+1,t+1})^{-\sigma}]\biggr]^{-\frac{1}{\sigma}}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): composite good price
        b_splus1 (array_like): household savings one period ahead
        n_splus1_policy (array_like): household labor supply one period
            ahead across b, z
        c_splus1_policy (array_like): household consumption one period
            ahead across b, z
        factor (scalar): scaling factor converting model units to dollars
        rho (array_like): mortality rates
        etr_params (list): parameters of the effective tax rate
            functions
        mtry_params (list): parameters of the marginal tax rate
            on capital income functions
        j (int): index of ability type
        t (int): model period
        e_splus1 (array_like): effective labor units one period ahead
        z_index (array_like): index in productivity grid
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    returns:
        c (array_like): household consumption in current period
    """
    beta = p.beta[j]
    # Simplified tax parameter setting
    if method == "SS":
        tax_noncompliance = p.capital_income_tax_noncompliance_rate[-1, j]
        income_tax_filer = p.income_tax_filer[-1, j]
        wealth_tax_filer = p.wealth_tax_filer[-1, j]
        h_wealth = p.h_wealth[-1]
        m_wealth = p.m_wealth[-1]
        p_wealth = p.p_wealth[-1]
    else:  # TPI
        tax_noncompliance = p.capital_income_tax_noncompliance_rate[t, j]
        income_tax_filer = p.income_tax_filer[t, j]
        wealth_tax_filer = p.wealth_tax_filer[t, j]
        h_wealth = p.h_wealth[t]
        m_wealth = p.m_wealth[t]
        p_wealth = p.p_wealth[t]

    bequest_utility = rho * marg_ut_beq(b_splus1, p.sigma, j, p)

    # Calculate the expectation of the discounted marginal utility of
    # consumption
    # n_splus1_policy must be [nb, nz], so we check dimensions.
    if n_splus1_policy.ndim < 2:
        # This case may occur if the next period is the last, handle simply.
        n_splus1_policy = np.tile(n_splus1_policy.reshape(-1, 1), (1, p.nz))
    consumption_utility_matrix = np.zeros(
        (b_splus1.shape[0], p.z_grid.shape[0])
    )
    for zp_index, zp in enumerate(p.z_grid):
        deriv = (
            (1 + r_splus1)
            - (
                r_splus1
                * tax.MTR_income(
                    r_splus1,
                    w_splus1,
                    b_splus1,
                    n_splus1_policy[:, zp_index],
                    factor,
                    True,
                    e_splus1 * zp,
                    etr_params,
                    mtry_params,
                    tax_noncompliance,
                    income_tax_filer,
                    p,
                )
            )
            - tax.MTR_wealth(
                b_splus1, h_wealth, m_wealth, p_wealth, wealth_tax_filer
            )
        )
        consumption_utility_matrix[:, zp_index] = (
            deriv
            * marg_ut_cons(c_splus1_policy[:, zp_index], p.sigma)
            / p_tilde_splus1
        )

    prob_z_splus1 = p.z_trans[z_index, :]
    E_MU_c = consumption_utility_matrix @ prob_z_splus1

    # Combine bequest utility and expectation of future consumption utility
    mu_c_rhs = bequest_utility + beta * (1 - rho) * E_MU_c

    # Final consumption calculation
    # c = inv_mu_c((p_tilde ** (p.sigma)) * mu_c_rhs, p.sigma)
    # is it correct to use p_tilde ** (p.sigma) here?
    c = np.exp(p.g_y) * inv_mu_c(p_tilde * mu_c_rhs, p.sigma)

    return c


def FOC_labor(
    r,
    w,
    p_tilde,
    b,
    c,
    n,
    factor,
    e,
    z,
    chi_n,
    etr_params,
    mtrx_params,
    t,
    j,
    p,
    method,
):
    r"""
    Computes errors for the FOC for labor supply in the steady
    state.  This function is usually looped through over J, so it does
    one lifetime income group at a time.

    .. math::
        w_t z e_{j,s}\bigl(1 - \tau^{mtrx}_{s,t}\bigr)
       \frac{(c_{j,s,t})^{-\sigma}}{ \tilde{p}_{t}} = \chi^n_{s}
        \biggl(\frac{b}{\tilde{l}}\biggr)\biggl(\frac{n_{j,s,t}}
        {\tilde{l}}\biggr)^{\upsilon-1}\Biggl[1 -
        \biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^\upsilon\Biggr]
        ^{\frac{1-\upsilon}{\upsilon}}

    Args:
        r (array_like): the real interest rate
        w (array_like): the real wage rate
        p_tilde (array_like): composite good price
        b (Numpy array): household savings
        b_splus1 (Numpy array): household savings one period ahead
        n (Numpy array): household labor supply
        bq (Numpy array): household bequests received
        factor (scalar): scaling factor converting model units to dollars
        tr (Numpy array): government transfers to household
        ubi (Numpy array): universal basic income payment
        theta (Numpy array): social security replacement rate for each
            lifetime income group
        chi_n (Numpy array): utility weight on the disutility of labor
            supply
        e (Numpy array): effective labor units
        etr_params (list): parameters of the effective tax rate
            functions
        mtrx_params (list): parameters of the marginal tax rate
            on labor income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        FOC_error (Numpy array): error from FOC for labor supply

    """
    if method == "SS":
        tau_payroll = p.tau_payroll[-1]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[-1, j]
        income_tax_filer = p.income_tax_filer[-1, j]
    elif method == "TPI_scalar":  # for 1st donut ring only
        tau_payroll = p.tau_payroll[0]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[0, j]
        income_tax_filer = p.income_tax_filer[0, j]
    else:
        tau_payroll = p.tau_payroll[t]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[t, j]
        income_tax_filer = p.income_tax_filer[t, j]

    deriv = (
        1
        - tau_payroll
        - tax.MTR_income(
            r,
            w,
            b,
            n,
            factor,
            False,
            e * z,
            etr_params,
            mtrx_params,
            tax_noncompliance,
            income_tax_filer,
            p,
        )
    )
    FOC_error = marg_ut_cons(c, p.sigma) * (
        1 / p_tilde
    ) * w * deriv * e * z - marg_ut_labor(n, chi_n, p)

    return FOC_error


def get_y(r_p, w, b_s, n, z, p, method):
    r"""
    Compute household income before taxes.

    .. math::
        y_{j,s,t} = r_{p,t}b_{j,s,t} + w_{t}e_{j,s}n_{j,s,t}

    Args:
        r_p (array_like): real interest rate on the household portfolio
        w (array_like): real wage rate
        b_s (Numpy array): household savings coming into the period
        n (Numpy array): household labor supply
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'
    """
    if method == "SS":
        e = np.squeeze(p.e[-1, :, :])
    elif method == "TPI":
        e = p.e
    y = r_p * b_s + w * e * z * n

    return y


def BC_residual(c, n, b, b_splus1, r, w, p_tilde, e, z, bq, net_tax, p):
    r"""
    Compute the residuals of the household budget constraint.

    .. math::
        c_{j,s,t} + b_{j,s+1,t+1} - (1 + r_{t})b_{j,s,t} =
        w_{t}e_{j,s}n_{j,s,t} + bq_{j,s,t} + tr_{j,s,t} - T_{j,s,t}
    """

    BC_error = (
        (1 + r) * b + w * e * z * n + bq - b_splus1 * np.exp(p.g_y) - net_tax
    ) - p_tilde * c
    return BC_error


def EOL_system(
    n,
    b,
    p_tilde,
    r,
    w,
    tr,
    ubi,
    bq,
    theta,
    factor,
    e,
    z,
    chi_n,
    etr_params,
    mtrx_params,
    t,
    j,
    p,
    method,
):
    r"""
    Compute the residuals of the household budget constraint at the end of
    life given a guess for labor supply. Solve first for consumption given
    labor supply and then for savings given consumption. Then check the
    budget constraint.

    Args:
        n (array_like): household labor supply
        b (array_like): household savings
        p_tilde (array_like): composite good price
        r (scalar): the real interest rate
        w (scalar): the real wage rate
        factor (scalar): scaling factor converting model units to dollars
        e (scalar): effective labor units
        z (scalar): productivity
        chi_n (scalar): utility weight on the disutility of labor
        etr_params (list): parameters of the effective tax rate functions
        mtrx_params (list): parameters of the marginal tax rate on labor
            income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        BC_error (float or array_like): residuals of the household budget
            constraint; a Python float when there is a single residual,
            as ``scipy.optimize.brentq`` requires
    """
    # change n to array
    n = np.atleast_1d(n)

    c = c_from_n(
        n,
        b,
        p_tilde,
        r,
        w,
        factor,
        e,
        z,
        chi_n,
        etr_params,
        mtrx_params,
        t,
        j,
        p,
        method,
    )
    # print("EOL c = ", c)
    # use consumption to get savings from savings Euler equation
    b_splus1 = b_from_c_EOL(c, p_tilde, j, p.sigma, p)
    net_tax = tax.net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        theta,
        t,
        j,
        False,
        method,
        e * z,
        etr_params,
        p,
    )

    BC_error = BC_residual(
        c, n, b, b_splus1, r, w, p_tilde, e, z, bq, net_tax, p
    )
    # brentq needs a scalar; numpy >= 2.3 will not convert a 1-element
    # array implicitly
    if np.size(BC_error) == 1:
        return float(np.squeeze(BC_error))
    return BC_error


def HH_system(
    x,
    c,
    b_splus1,
    r,
    w,
    p_tilde,
    factor,
    tr,
    ubi,
    bq,
    theta,
    e,
    z,
    chi_n,
    etr_params,
    mtrx_params,
    j,
    t,
    p,
    method,
):
    r"""
    Compute the residuals of the household budget constraint and labor
    supply Euler equation given a guess of household assets and labor
    choice. This is for use in a root finder to solve the household
    problem at age s < E+S.

    Args:
        x (array_like): vector containing household assets b and labor supply n
        c (array_like): household consumption
        b_splus1 (array_like): household savings one period ahead
        r (scalar): the real interest rate
        w (scalar): the real wage rate
        p_tilde (scalar): composite good price
        factor (scalar): scaling factor converting model units to dollars
        e (scalar): effective labor units
        z (scalar): productivity
        chi_n (scalar): utility weight on the disutility of labor
        etr_params (list): parameters of the effective tax rate functions
        mtrx_params (list): parameters of the marginal tax rate on labor
            income functions
        j (int): index of ability type
        t (int): model period
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        HH_error (array_like): residuals of the household budget
            constraint and labor supply Euler equation
    """
    b = x[0]
    n = np.atleast_1d(x[1])
    net_tax = tax.net_taxes(
        r,
        w,
        b,
        n,
        bq,
        factor,
        tr,
        ubi,
        theta,
        t,
        j,
        False,
        method,
        e * z,
        etr_params,
        p,
    )
    BC_error = BC_residual(
        c, n, b, b_splus1, r, w, p_tilde, e, z, bq, net_tax, p
    )
    FOC_error = FOC_labor(
        r,
        w,
        p_tilde,
        b,
        c,
        n,
        factor,
        e,
        z,
        chi_n,
        etr_params,
        mtrx_params,
        t,
        j,
        p,
        method,
    )
    HH_error = np.array([BC_error, FOC_error])
    return np.squeeze(HH_error)


def solve_HH(
    r,
    w,
    p_tilde,
    factor,
    tr,
    bq,
    ubi,
    b_grid,
    sigma,
    theta,
    chi_n,
    rho,
    e,
    etr_params,
    mtrx_params,
    mtry_params,
    j,
    t,
    p,
    method,
):
    # solve household problem on transition path using endogenous grid method
    nb = len(b_grid)

    # if method == "SS":
    #    r = np.repeat(r, p.S)
    #    w = np.repeat(w, p.S)
    #    p_tilde = np.repeat(p_tilde, p.S)

    # initialize policy functions on grid
    b_policy = np.zeros((p.S, nb, p.nz))
    c_policy = np.zeros((p.S, nb, p.nz))
    n_policy = np.zeros((p.S, nb, p.nz))

    # start at the end of life
    s_EOL = p.S - 1
    current_t = t[s_EOL] if hasattr(t, "__len__") else t
    ubi_sEOL = (
        ubi[s_EOL] if isinstance(ubi, np.ndarray) and ubi.size == p.S else ubi
    )
    theta_sEOL = (
        theta[s_EOL]
        if isinstance(theta, np.ndarray) and theta.size == p.S
        else theta
    )
    for z_index, z in enumerate(p.z_grid):
        for b_index, b in enumerate(b_grid):
            # use root finder to solve problem at end of life
            args = (
                b,
                p_tilde[s_EOL],
                r[s_EOL],
                w[s_EOL],
                tr[s_EOL],
                ubi_sEOL,
                bq[s_EOL],
                theta_sEOL,
                factor,
                e[s_EOL],
                z,
                chi_n[s_EOL],
                etr_params[s_EOL, :],
                mtrx_params[s_EOL, :],
                current_t,  # Handle scalar t
                j,
                p,
                method,
            )
            n = opt.brentq(EOL_system, 1e-8, p.ltilde - 1e-8, args=args)
            n_policy[-1, b_index, z_index] = n
            c_policy[-1, b_index, z_index] = c_from_n(
                n,
                b,
                p_tilde[s_EOL],
                r[s_EOL],
                w[s_EOL],
                factor,
                e[s_EOL],
                z,
                chi_n[s_EOL],
                etr_params[s_EOL, :],
                mtrx_params[s_EOL, :],
                current_t,  # Handle scalar t
                j,
                p,
                method,
            )
            b_policy[-1, b_index, z_index] = b_from_c_EOL(
                c_policy[-1, b_index, z_index], p_tilde[-1], j, p.sigma, p
            )

    # iterate backwards with Euler equation
    for s in range(p.S - 2, -1, -1):
        ubi_s = (
            ubi[s] if isinstance(ubi, np.ndarray) and ubi.size == p.S else ubi
        )
        theta_s = (
            theta[s]
            if isinstance(theta, np.ndarray) and theta.size == p.S
            else theta
        )
        for z_index, z in enumerate(
            p.z_grid
        ):  # NOTE: Maybe able to parallelize this loop
            c = c_from_b_splus1(
                r[s + 1],
                w[s + 1],
                p_tilde[s + 1],
                p_tilde[s],
                b_grid,
                n_policy[s + 1, :, :],
                c_policy[s + 1, :, :],
                factor,
                rho[s],
                etr_params[s + 1, :],
                mtry_params[s + 1, :],
                j,
                t[s + 1] if hasattr(t, "__len__") else t,
                e[s + 1],
                z_index,
                p,
                method,
            )
            b = np.zeros_like(b_grid)
            n = np.zeros_like(b_grid)
            current_t = t[s] if hasattr(t, "__len__") else t
            for b_splus1_index, b_splus1 in enumerate(
                b_grid
            ):  # NOTE: may be able to parallelize this loop (or one above)
                args = (
                    c[b_splus1_index],
                    b_splus1,
                    r[s],
                    w[s],
                    p_tilde[s],
                    factor,
                    tr[s],
                    ubi_s,
                    bq[s],
                    theta_s,  # [s],
                    e[s],
                    z,
                    chi_n[s],
                    etr_params[s, :],
                    mtrx_params[s, :],
                    j,
                    current_t,
                    p,
                    method,
                )
                initial_guess = np.array(
                    [b_splus1, n_policy[s + 1, b_splus1_index, z_index]]
                )
                # Fall back to the initial guess if root-finding fails
                res = opt.root(
                    HH_system, initial_guess, args=args, method="lm"
                )
                if res.success:
                    b[b_splus1_index], n[b_splus1_index] = res.x
                else:  # Handle failure, e.g., by using the guess
                    print(
                        f"Root finding failed for b_splus1={b_splus1} "
                        f"at s={s}, z_index={z_index}."
                    )
                    b[b_splus1_index], n[b_splus1_index] = initial_guess

            # clean for monotonicity
            b_clean, unique_idx = np.unique(b, return_index=True)
            c_interp = c[unique_idx]
            n_interp = n[unique_idx]
            b_splus1_interp = b_grid[unique_idx]

            # use scipy pchip interpolation
            c_itp = itp.PchipInterpolator(b_clean, c_interp, extrapolate=True)
            n_itp = itp.PchipInterpolator(b_clean, n_interp, extrapolate=True)
            b_splus1_itp = itp.PchipInterpolator(
                b_clean, b_splus1_interp, extrapolate=True
            )
            c_policy[s, :, z_index] = c_itp(b_grid)
            n_policy[s, :, z_index] = n_itp(b_grid)
            b_policy[s, :, z_index] = b_splus1_itp(b_grid)

    print("Labor supply for young with no assets:", n_policy[0, 0, :])
    print("Labor supply for young with little assets:", n_policy[0, 1, :])

    return b_policy, c_policy, n_policy


def solve_all_households_ss(r, w, p_tilde, factor, TR, BQ, p, b_grid):
    """
    Solves for the steady-state policy functions for all household types j.

    This function iterates through each ability type `j`, calculates the
    type-specific transfers and bequests, and then calls the household
    solver `household_stoch.solve_HH` to find the optimal policy functions
    for savings, consumption, and labor supply over an asset grid and
    for each productivity shock.

    Args:
        r (scalar): The steady-state real interest rate.
        w (scalar): The steady-state real wage rate.
        p_tilde (scalar): The steady-state composite good price.
        factor (scalar): The steady-state scaling factor.
        TR (scalar): Aggregate government transfers in the steady state.
        BQ (array_like): Aggregate bequests for each ability type (J,).
        p (Specifications): An object containing all model parameters.
        b_grid (array): The grid of assets over which to solve the
            household problem, shape (nb,).

    Returns:
        tuple: A tuple containing the policy functions and the asset grid:
            - b_policy (ndarray): Savings policy, shape (S, J, nb, nz).
            - c_policy (ndarray): Consumption policy, shape (S, J, nb, nz).
            - n_policy (ndarray): Labor supply policy, shape (S, J, nb, nz).
            - b_grid (ndarray): The grid of assets, shape (nb,).
    """
    nb = len(b_grid)

    # Initialize policy function arrays to store results for all J types
    b_policy_all = np.zeros((p.S, p.J, nb, p.nz))
    c_policy_all = np.zeros((p.S, p.J, nb, p.nz))
    n_policy_all = np.zeros((p.S, p.J, nb, p.nz))
    ss_etr_params = np.array(p.etr_params)[-1, :, :]
    ss_mtrx_params = np.array(p.mtrx_params)[-1, :, :]
    ss_mtry_params = np.array(p.mtry_params)[-1, :, :]

    # Loop over each ability type
    for j in range(p.J):  # CAN PARALLELIZE HERE
        # --- Prepare arguments for the household solver for type j ---

        # Create S-length vectors for time-varying parameters (constant in SS)
        ss_r = np.ones(p.S) * r
        ss_w = np.ones(p.S) * w
        ss_p_tilde = np.ones(p.S) * p_tilde

        # Calculate type- and age-specific transfers and bequests
        # get_bq and get_tr come from ogcore.household
        tr_j = get_tr(TR, j, p, "SS")
        bq_j = get_bq(BQ, j, p, "SS")
        ubi_j = p.ubi_nom_array[-1, :, j] / factor

        # Extract SS parameters for the current type j
        # [-1] is to get the steady-state value from time-path arrays
        e_j = p.e[-1, :, j]
        chi_n = p.chi_n[-1, :]

        # For SS, the time index `t` is not advancing
        ss_t = np.zeros(p.S, dtype=int)

        # In the stochastic model provided, theta (pension replacement rate) is
        # not endogenously calculated. Following the test files, we set it
        # to zero.
        theta_j = np.zeros(p.S)

        # Assemble the dictionary of arguments for the solver
        solver_args = {
            "r": ss_r,
            "w": ss_w,
            "p_tilde": ss_p_tilde,
            "factor": factor,
            "tr": tr_j,
            "bq": bq_j,
            "ubi": ubi_j,
            "b_grid": b_grid,
            "sigma": p.sigma,
            "theta": theta_j,
            "chi_n": chi_n,
            "rho": p.rho[-1, :, j],
            "e": e_j,
            "etr_params": ss_etr_params,
            "mtrx_params": ss_mtrx_params,
            "mtry_params": ss_mtry_params,
            "j": j,
            "t": ss_t,
            "p": p,
            "method": "SS",
        }

        # --- Solve the household problem for type j ---
        print(f"Solving for household type j={j}...")
        b_policy_j, c_policy_j, n_policy_j = solve_HH(**solver_args)
        print(f"Finished solving for type j={j}.")

        # --- Store the results ---
        b_policy_all[:, j, :, :] = b_policy_j
        c_policy_all[:, j, :, :] = c_policy_j
        n_policy_all[:, j, :, :] = n_policy_j

    return b_policy_all, c_policy_all, n_policy_all


def get_ss_distribution(p, b_policy, b_grid, initial_dist=None):
    """
    Calculates the steady-state distribution of agents over age,
    ability type, assets, and productivity shocks using Young's method.

    This function iterates forward from an initial distribution of
    newborns, distributing the mass of agents across the asset grid
    in the next period according to the savings policy function.

    Args:
        p (Specifications): OG-Core Specifications object containing model
            parameters.
        b_policy (Numpy array): The optimal savings policy function,
            with shape (S, J, nb, nz).
        b_grid (Numpy array): The grid of asset values, with shape (nb,).
        initial_dist (Numpy array, optional): The initial distribution
            of agents at age s=0. It should have shape (J, nb, nz).
            If None, a default is used where newborns start at the
            lowest asset level, distributed by ability type according
            to p.lambdas and by productivity according to the stationary
            distribution of p.z_trans. Defaults to None.

    Returns:
        dist (Numpy array): The steady-state distribution of agents,
            with shape (S, J, nb, nz). The distribution sums to 1.
    """
    # Get key parameters and dimensions from the Specifications object
    S, J, nz = p.S, p.J, p.nz
    nb = len(b_grid)
    lambdas = p.lambdas
    z_trans = p.z_trans
    b_min, b_max = b_grid[0], b_grid[-1]

    # Initialize the distribution array
    dist = np.zeros((S, J, nb, nz))

    # Set the initial distribution for newborns (age s=0)
    if initial_dist is not None:
        # Use the provided initial distribution
        # expected_shape = (J, nb, nz)
        # if initial_dist.shape != expected_shape:
        #    raise ValueError(
        #        f"The provided initial_dist has shape {initial_dist.shape} "
        #        f"but must have shape {expected_shape}."
        #    )
        dist[0, :, :, :] = initial_dist
    else:
        # if none, newborns start at the lowest asset level
        # with a stationary distribution of productivity shocks.
        # Find the eigenvector of z_trans.T corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(z_trans.T)
        unit_eigenvalue_idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary_dist_z = np.real(eigenvectors[:, unit_eigenvalue_idx])

        # Normalize the eigenvector to sum to 1
        stationary_dist_z /= stationary_dist_z.sum()
        for j in range(J):
            dist[0, j, 0, :] = lambdas[j] * stationary_dist_z

    # Iterate forward through ages to calculate the distribution
    for s in range(S - 1):
        for j in range(J):
            for b_idx in range(nb):
                for z_idx in range(nz):
                    # If there's no mass at this point, skip it
                    if dist[s, j, b_idx, z_idx] == 0:
                        continue

                    # Get next period's savings from the policy function
                    b_next = b_policy[s, j, b_idx, z_idx]

                    # Distribute the mass to the next age cohort (s+1)
                    # based on the productivity shock transition matrix
                    for z_next_idx in range(nz):
                        mass_to_move = (
                            dist[s, j, b_idx, z_idx]
                            * z_trans[z_idx, z_next_idx]
                        )

                        # Find where b_next falls on the asset grid and
                        # distribute the mass accordingly.
                        if b_next <= b_min:
                            # Entire mass goes to the lowest grid point
                            dist[s + 1, j, 0, z_next_idx] += mass_to_move
                        elif b_next >= b_max:
                            # Entire mass goes to the highest grid point
                            dist[s + 1, j, -1, z_next_idx] += mass_to_move
                        else:
                            # Use linear interpolation (histogram method) to
                            # split the mass between the two nearest grid
                            # points.
                            # Find the index of the grid point to the right
                            next_b_idx = np.searchsorted(b_grid, b_next)

                            # Calculate the weight for the higher grid point
                            weight_high = (b_next - b_grid[next_b_idx - 1]) / (
                                b_grid[next_b_idx] - b_grid[next_b_idx - 1]
                            )
                            # The weight for the lower point is the remainder
                            weight_low = 1 - weight_high

                            # Update the distribution for the next period
                            dist[s + 1, j, next_b_idx - 1, z_next_idx] += (
                                mass_to_move * weight_low
                            )
                            dist[s + 1, j, next_b_idx, z_next_idx] += (
                                mass_to_move * weight_high
                            )

    return dist
