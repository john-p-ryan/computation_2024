"""
------------------------------------------------------------------------
Household functions.
------------------------------------------------------------------------
"""

# Packages
import numpy as np
import scipy.interpolate as itp
import scipy.optimize as opt
from ogcore import tax, utils

"""
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
"""


def marg_ut_cons(c, sigma):
    r"""
    Compute the marginal utility of consumption.

    .. math::
        MU_{c} = c^{-\sigma}

    Args:
        c (array_like): household consumption
        sigma (scalar): coefficient of relative risk aversion

    Returns:
        output (array_like): marginal utility of consumption

    """
    if np.ndim(c) == 0:
        c = np.array([c])
    epsilon = 0.003
    cvec_cnstr = c < epsilon
    MU_c = np.zeros(c.shape)
    MU_c[~cvec_cnstr] = c[~cvec_cnstr] ** (-sigma)
    b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
    b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
    MU_c[cvec_cnstr] = 2 * b2 * c[cvec_cnstr] + b1
    output = MU_c
    output = np.squeeze(output)

    return output


def marg_ut_labor(n, chi_n, p):
    r"""
    Compute the marginal disutility of labor.

    .. math::
        MDU_{l} = \chi^n_{s}\biggl(\frac{b}{\tilde{l}}\biggr)
        \biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^{\upsilon-1}
        \Biggl[1-\biggl(\frac{n_{j,s,t}}{\tilde{l}}\biggr)^\upsilon
        \Biggr]^{\frac{1-\upsilon}{\upsilon}}

    Args:
        n (array_like): household labor supply
        chi_n (array_like): utility weights on disutility of labor
        p (OG-Core Specifications object): model parameters

    Returns:
        output (array_like): marginal disutility of labor supply

    """
    # print("In marg_ut_labor")
    # print("n = ", n)
    # print("params = ", p.b_ellipse, p.ltilde, p.upsilon, chi_n)
    nvec = n
    if np.ndim(nvec) == 0:
        nvec = np.array([nvec])
    eps_low = 0.000001
    eps_high = p.ltilde - 0.000001
    nvec_low = nvec < eps_low
    nvec_high = nvec > eps_high
    nvec_uncstr = np.logical_and(~nvec_low, ~nvec_high)
    MDU_n = np.zeros(nvec.shape)
    MDU_n[nvec_uncstr] = (
        (p.b_ellipse / p.ltilde)
        * ((nvec[nvec_uncstr] / p.ltilde) ** (p.upsilon - 1))
        * (
            (1 - ((nvec[nvec_uncstr] / p.ltilde) ** p.upsilon))
            ** ((1 - p.upsilon) / p.upsilon)
        )
    )
    b2 = (
        0.5
        * p.b_ellipse
        * (p.ltilde ** (-p.upsilon))
        * (p.upsilon - 1)
        * (eps_low ** (p.upsilon - 2))
        * (
            (1 - ((eps_low / p.ltilde) ** p.upsilon))
            ** ((1 - p.upsilon) / p.upsilon)
        )
        * (
            1
            + ((eps_low / p.ltilde) ** p.upsilon)
            * ((1 - ((eps_low / p.ltilde) ** p.upsilon)) ** (-1))
        )
    )
    b1 = (p.b_ellipse / p.ltilde) * (
        (eps_low / p.ltilde) ** (p.upsilon - 1)
    ) * (
        (1 - ((eps_low / p.ltilde) ** p.upsilon))
        ** ((1 - p.upsilon) / p.upsilon)
    ) - (
        2 * b2 * eps_low
    )
    MDU_n[nvec_low] = 2 * b2 * nvec[nvec_low] + b1
    d2 = (
        0.5
        * p.b_ellipse
        * (p.ltilde ** (-p.upsilon))
        * (p.upsilon - 1)
        * (eps_high ** (p.upsilon - 2))
        * (
            (1 - ((eps_high / p.ltilde) ** p.upsilon))
            ** ((1 - p.upsilon) / p.upsilon)
        )
        * (
            1
            + ((eps_high / p.ltilde) ** p.upsilon)
            * ((1 - ((eps_high / p.ltilde) ** p.upsilon)) ** (-1))
        )
    )
    d1 = (p.b_ellipse / p.ltilde) * (
        (eps_high / p.ltilde) ** (p.upsilon - 1)
    ) * (
        (1 - ((eps_high / p.ltilde) ** p.upsilon))
        ** ((1 - p.upsilon) / p.upsilon)
    ) - (
        2 * d2 * eps_high
    )
    MDU_n[nvec_high] = 2 * d2 * nvec[nvec_high] + d1
    output = MDU_n * np.squeeze(chi_n)
    output = np.squeeze(output)
    return output


def marg_ut_beq(b, sigma, j, p):
    r"""
    Compute the marginal utility of savings.

    .. math::
        MU_{b} = \chi^b_{j}b_{j,s,t}^{-\sigma}

    Args:
        b (array_like): household savings
        chi_b (array_like): utility weights on savings
        p (OG-Core Specifications object): model parameters

    Returns:
        output (array_like): marginal utility of savings

    """
    if np.ndim(b) == 0:
        b = np.array([b])
    epsilon = 0.0001
    bvec_cnstr = b < epsilon
    MU_b = np.zeros(b.shape)
    MU_b[~bvec_cnstr] = p.chi_b[j] * b[~bvec_cnstr] ** (-sigma)
    b2 = (-sigma * (epsilon ** (-sigma - 1))) / 2
    b1 = (epsilon ** (-sigma)) - 2 * b2 * epsilon
    MU_b[bvec_cnstr] = 2 * b2 * b[bvec_cnstr] + b1
    output = MU_b
    output = np.squeeze(output)
    return output


def inv_mu_c(value, sigma):
    r"""
    Compute the inverse of the marginal utility of consumption.

    .. math::
    c = \left(\frac{1}{val}\right)^{-1/\sigma}

    Args:
    value (array_like): marginal utility of consumption
    sigma (scalar): coefficient of relative risk aversion

    Returns:
    output (array_like): household consumption

    """
    if np.ndim(value) == 0:
        value = np.array([value])
    epsilon = .0001
    value_cnstr = value < epsilon
    output = np.zeros(value.shape)
    output[~value_cnstr] = value[~value_cnstr] ** (-1 / sigma)
    # where value is constrained, we use a Taylor expansion:
    # c = epsilon ** (-1 / sigma) - 1/sigma * epsilon ** (-(1+sigma)/sigma) * (value - epsilon)
    b2 = (-1 / sigma) * (epsilon ** (-(1 + sigma) / sigma))
    b1 = epsilon ** (-1 / sigma)
    output[value_cnstr] = b1 + b2 * (value[value_cnstr] - epsilon)
    output = np.squeeze(output)
    return output


def get_bq(BQ, j, p, method):
    r"""
    Calculate bequests to each household.

    .. math::
        bq_{j,s,t} = \zeta_{j,s}\frac{BQ_{t}}{\lambda_{j}\omega_{s,t}}

    Args:
        BQ (array_like): aggregate bequests
        j (int): index of lifetime ability group
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        bq (array_like): bequests received by each household

    """
    if p.use_zeta:
        if j is not None:
            if method == "SS":
                bq = (p.zeta[:, j] * BQ) / (p.lambdas[j] * p.omega_SS)
            else:
                len_T = BQ.shape[0]
                bq = (
                    np.reshape(p.zeta[:, j], (1, p.S)) * BQ.reshape((len_T, 1))
                ) / (p.lambdas[j] * p.omega[:len_T, :])
        else:
            if method == "SS":
                bq = (p.zeta * BQ) / (
                    p.lambdas.reshape((1, p.J)) * p.omega_SS.reshape((p.S, 1))
                )
            else:
                len_T = BQ.shape[0]
                bq = (
                    np.reshape(p.zeta, (1, p.S, p.J))
                    * utils.to_timepath_shape(BQ)
                ) / (
                    p.lambdas.reshape((1, 1, p.J))
                    * p.omega[:len_T, :].reshape((len_T, p.S, 1))
                )
    else:
        if j is not None:
            if method == "SS":
                bq = np.tile(BQ[j], p.S) / p.lambdas[j]
            if method == "TPI":
                len_T = BQ.shape[0]
                bq = np.tile(
                    np.reshape(BQ[:, j] / p.lambdas[j], (len_T, 1)), (1, p.S)
                )
        else:
            if method == "SS":
                BQ_per = BQ / np.squeeze(p.lambdas)
                bq = np.tile(np.reshape(BQ_per, (1, p.J)), (p.S, 1))
            if method == "TPI":
                len_T = BQ.shape[0]
                BQ_per = BQ / p.lambdas.reshape(1, p.J)
                bq = np.tile(np.reshape(BQ_per, (len_T, 1, p.J)), (1, p.S, 1))
    return bq


def get_tr(TR, j, p, method):
    r"""
    Calculate transfers to each household.

    .. math::
        tr_{j,s,t} = \zeta_{j,s}\frac{TR_{t}}{\lambda_{j}\omega_{s,t}}

    Args:
        TR (array_like): aggregate transfers
        j (int): index of lifetime ability group
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or
            'TPI'

    Returns:
        tr (array_like): transfers received by each household

    """
    if j is not None:
        if method == "SS":
            tr = (p.eta[-1, :, j] * TR) / (p.lambdas[j] * p.omega_SS)
        else:
            len_T = TR.shape[0]
            tr = (p.eta[:len_T, :, j] * TR.reshape((len_T, 1))) / (
                p.lambdas[j] * p.omega[:len_T, :]
            )
    else:
        if method == "SS":
            tr = (p.eta[-1, :, :] * TR) / (
                p.lambdas.reshape((1, p.J)) * p.omega_SS.reshape((p.S, 1))
            )
        else:
            len_T = TR.shape[0]
            tr = (p.eta[:len_T, :, :] * utils.to_timepath_shape(TR)) / (
                p.lambdas.reshape((1, 1, p.J))
                * p.omega[:len_T, :].reshape((len_T, p.S, 1))
            )

    return tr


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
    Calculate household consumption from labor supply Euler equation for group j.

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
        if e is None:
            e = np.squeeze(p.e[-1, :, j])
    elif method == "TPI":
        # When t is a scalar, we are solving for a particular period
        tau_payroll = p.tau_payroll[t]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[t, j]
        if e is None:
            # Simplified for TPI, assuming e is passed correctly for age s
            # The original logic was overly complex for this function's scope
            e = p.e[t, :, j]
    else:  # Fallback for other methods if needed
        tau_payroll = p.tau_payroll[0]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[0, j]

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
            p,
        )
    )
    numerator = (
        p_tilde * marg_ut_labor(n, chi_n, p)
    )
    denominator = w * e * z * deriv
    # print("numerator:", numerator,  marg_ut_labor(n, chi_n, p), (1 - p.sigma))
    # print("denominator:", denominator, deriv)
    c = inv_mu_c(numerator / denominator, p.sigma)

    return c


'''
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
    Calculate household consumption from labor supply Euler equation for group j.

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
    if method == "SS":
        tau_payroll = p.tau_payroll[-1]
    elif method == "TPI_scalar":  # for 1st donut ring only
        tau_payroll = p.tau_payroll[0]
    else:
        length = r.shape[0]
        tau_payroll = p.tau_payroll[t : t + length]
    if j is not None:
        if method == "SS":
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[-1, j]
            if e is None:
                e = np.squeeze(p.e[-1, :, j])
        elif method == "TPI_scalar":
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[0, j]
            if e is None:
                e = np.squeeze(p.e[0, -1, j])
        else:
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[
                t : t + length, j
            ]
            if e is None:
                e_long = np.concatenate(
                    (
                        p.e,
                        np.tile(p.e[-1, :, :].reshape(1, p.S, p.J), (p.S, 1, 1)),
                    ),
                    axis=0,
                )
                e = np.diag(e_long[t : t + p.S, :, j], max(p.S - length, 0))
    else:
        if method == "SS":
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[-1, :]
            if e is None:
                e = np.squeeze(p.e[-1, :, :])
        elif method == "TPI_scalar":
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[0, :]
            if e is None:
                e = np.squeeze(p.e[0, -1, :])
        else:
            tax_noncompliance = p.labor_income_tax_noncompliance_rate[
                t : t + length, :
            ]
            if e is None:
                e_long = np.concatenate(
                    (
                        p.e,
                        np.tile(p.e[-1, :, :].reshape(1, p.S, p.J), (p.S, 1, 1)),
                    ),
                    axis=0,
                )
                e = np.diag(e_long[t : t + p.S, :, j], max(p.S - length, 0))
    if method == "SS":
        tau_payroll = p.tau_payroll[-1]
    elif method == "TPI_scalar":  # for 1st donut ring only
        tau_payroll = p.tau_payroll[0]
    else:
        length = r.shape[0]
        tau_payroll = p.tau_payroll[t : t + length]
    if method == "TPI":
        if b.ndim == 2:
            r = r.reshape(r.shape[0], 1)
            w = w.reshape(w.shape[0], 1)
            tau_payroll = tau_payroll.reshape(tau_payroll.shape[0], 1)

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
            p,
        )
    )
    numerator = (
        p_tilde
        * np.exp(p.g_y * (1 - p.sigma))
        * marg_ut_labor(n, chi_n, p)
    )
    denominator = w * e * z * deriv
    c = inv_mu_c(numerator / denominator, p.sigma)

    return c
'''


def b_from_c_EOL(c, p_tilde, j, sigma, p):
    r"""
    Calculate household bequests at the end of life from the savings Euler equation.

    .. math::
        b_{j, E+S+1, t+1} = [\chi_j^b \tilde p_t]^{\frac{1}{\sigma}} * c_{j, E+S, t}

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


def euler_error(
    c_s,
    b_splus1,
    r_splus1,
    w_splus1,
    p_tilde_splus1,
    p_tilde_s,
    n_splus1_interp,
    c_splus1_interp,
    factor,
    rho,
    etr_params_splus1,
    mtry_params_splus1,
    j,
    t_splus1,
    e_splus1,
    z_index,
    p,
    method,
):
    r"""
    Calculates the error in the Euler equation for savings.
    This function computes the difference between the marginal utility of
    consumption today and the expected discounted marginal utility of
    consumption tomorrow, including bequests.

    .. math::
        \frac{u'(c_s)}{\tilde{p}_s} = \rho_s u'_b(b_{s+1}) +
        \beta(1-\rho_s)e^{g_y(1-\sigma)} E_s \Big[
        \frac{R_{s+1} u'(c_{s+1})}{\tilde{p}_{s+1}} \Big]

    Args:
        c_s (array_like): Consumption in the current period (s).
        b_splus1 (array_like): Savings for the next period (s+1).
        r_splus1 (scalar): Interest rate in period s+1.
        w_splus1 (scalar): Wage rate in period s+1.
        p_tilde_splus1 (scalar): Composite good price in period s+1.
        p_tilde_s (scalar): Composite good price in period s.
        n_splus1_interp (list): List of interpolation functions for labor
            supply in s+1, one for each z state.
        c_splus1_interp (list): List of interpolation functions for consumption
            in s+1, one for each z state.
        factor (scalar): scaling factor converting model units to dollars
        rho (scalar): mortality rate for age s
        etr_params_splus1 (array_like): ETR parameters for period s+1
        mtry_params_splus1 (array_like): MTRy parameters for period s+1
        j (int): index of ability type
        t_splus1 (int): model period for s+1
        e_splus1 (scalar): deterministic effective labor units for s+1
        z_index (int): index of current period stochastic productivity shock
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        euler_err (array_like): The error in the Euler equation.
    """
    beta = p.beta[j]
    if method == "SS":
        tax_noncompliance = p.capital_income_tax_noncompliance_rate[-1, j]
        h_wealth = p.h_wealth[-1]
        m_wealth = p.m_wealth[-1]
        p_wealth = p.p_wealth[-1]
    else:  # TPI
        tax_noncompliance = p.capital_income_tax_noncompliance_rate[t_splus1, j]
        h_wealth = p.h_wealth[t_splus1]
        m_wealth = p.m_wealth[t_splus1]
        p_wealth = p.p_wealth[t_splus1]

    # Marginal utility of consumption today (LHS of Euler equation)
    lhs = marg_ut_cons(c_s, p.sigma) / p_tilde_s

    # RHS of Euler equation
    bequest_utility = rho * marg_ut_beq(b_splus1, p.sigma, j, p)

    b_splus1_flat = np.atleast_1d(b_splus1)
    E_MU_c = np.zeros_like(b_splus1_flat)

    for zp_index, zp in enumerate(p.z_grid):
        n_splus1 = n_splus1_interp[zp_index](b_splus1_flat)
        c_splus1 = c_splus1_interp[zp_index](b_splus1_flat)

        # After-tax return on savings
        deriv = (1 + r_splus1) - (
            r_splus1
            * tax.MTR_income(
                r_splus1,
                w_splus1,
                b_splus1_flat,
                n_splus1,
                factor,
                True,
                e_splus1 * zp,
                etr_params_splus1,
                mtry_params_splus1,
                tax_noncompliance,
                p,
            )
        ) - tax.MTR_wealth(b_splus1_flat, h_wealth, m_wealth, p_wealth)

        # Marginal utility of consumption in s+1, weighted by return and price
        MU_c_splus1 = marg_ut_cons(c_splus1, p.sigma)
        term = deriv * MU_c_splus1 / p_tilde_splus1

        # Probability of transitioning to this z state
        prob_z_splus1 = p.Z[z_index, zp_index]
        E_MU_c += prob_z_splus1 * term

    growth_term = np.exp(p.g_y * (1 - p.sigma))
    rhs = bequest_utility + beta * (1 - rho) * growth_term * E_MU_c

    error = lhs - rhs
    return np.squeeze(error)


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


def get_ci(c_s, p_i, p_tilde, tau_c, alpha_c, method="SS"):
    r"""
    Compute consumption of good i given amount of composite consumption
    and prices.

    .. math::
        c_{i,j,s,t} = \frac{c_{s,j,t}}{\alpha_{i,j}p_{i,j}}

    Args:
        c_s (array_like): composite consumption
        p_i (array_like): prices for consumption good i
        p_tilde (array_like): composite good price
        tau_c (array_like): consumption tax rate
        alpha_c (array_like): consumption share parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

    Returns:
        c_si (array_like): consumption of good i
    """
    if method == "SS":
        I = alpha_c.shape[0]
        S = c_s.shape[0]
        J = c_s.shape[1]
        tau_c = tau_c.reshape(I, 1, 1)
        alpha_c = alpha_c.reshape(I, 1, 1)
        p_tilde.reshape(1, 1, 1)
        p_i = p_i.reshape(I, 1, 1)
        c_s = c_s.reshape(1, S, J)
        c_si = alpha_c * (((1 + tau_c) * p_i) / p_tilde) ** (-1) * c_s
    else:  # Time path case
        I = alpha_c.shape[0]
        T = p_i.shape[0]
        S = c_s.shape[1]
        J = c_s.shape[2]
        tau_c = tau_c.reshape(T, I, 1, 1)
        alpha_c = alpha_c.reshape(1, I, 1, 1)
        p_tilde = p_tilde.reshape(T, 1, 1, 1)
        p_i = p_i.reshape(T, I, 1, 1)
        c_s = c_s.reshape(T, 1, S, J)
        c_si = alpha_c * (((1 + tau_c) * p_i) / p_tilde) ** (-1) * c_s
    return c_si




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
    elif method == "TPI_scalar":  # for 1st donut ring only
        tau_payroll = p.tau_payroll[0]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[0, j]
    else:
        tau_payroll = p.tau_payroll[t]
        tax_noncompliance = p.labor_income_tax_noncompliance_rate[t, j]

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


def constraint_checker_SS(bssmat, nssmat, cssmat, ltilde):
    """
    Checks constraints on consumption, savings, and labor supply in the
    steady state.

    Args:
        bssmat (Numpy array): steady state distribution of capital
        nssmat (Numpy array): steady state distribution of labor
        cssmat (Numpy array): steady state distribution of consumption
        ltilde (scalar): upper bound of household labor supply

    Returns:
        None

    Raises:
        Warnings: if constraints are violated, warnings printed

    """
    print("Checking constraints on capital, labor, and consumption.")

    if (bssmat < 0).any():
        print("\tWARNING: There is negative capital stock")
    flag2 = False
    if (nssmat < 0).any():
        print(
            "\tWARNING: Labor supply violates nonnegativity ", "constraints."
        )
        flag2 = True
    if (nssmat > ltilde).any():
        print("\tWARNING: Labor supply violates the ltilde constraint.")
        flag2 = True
    if flag2 is False:
        print(
            "\tThere were no violations of the constraints on labor",
            " supply.",
        )
    if (cssmat < 0).any():
        print("\tWARNING: Consumption violates nonnegativity", " constraints.")
    else:
        print(
            "\tThere were no violations of the constraints on", " consumption."
        )


def constraint_checker_TPI(b_dist, n_dist, c_dist, t, ltilde):
    """
    Checks constraints on consumption, savings, and labor supply along
    the transition path. Does this for each period t separately.

    Args:
        b_dist (Numpy array): distribution of capital at time t
        n_dist (Numpy array): distribution of labor at time t
        c_dist (Numpy array): distribution of consumption at time t
        t (int): time period
        ltilde (scalar): upper bound of household labor supply

    Returns:
        None

    Raises:
        Warnings: if constraints are violated, warnings printed

    """
    if (b_dist <= 0).any():
        print(
            "\tWARNING: Aggregate capital is less than or equal to ",
            "zero in period %.f." % t,
        )
    if (n_dist < 0).any():
        print(
            "\tWARNING: Labor supply violates nonnegativity",
            " constraints in period %.f." % t,
        )
    if (n_dist > ltilde).any():
        print(
            "\tWARNING: Labor suppy violates the ltilde constraint",
            " in period %.f." % t,
        )
    if (c_dist < 0).any():
        print(
            "\tWARNING: Consumption violates nonnegativity",
            " constraints in period %.f." % t,
        )


def BC_residual(c, n, b, b_splus1, r, w, p_tilde, e, z, bq, net_tax, p):
    r"""
    Compute the residuals of the household budget constraint.

    .. math::
        c_{j,s,t} + b_{j,s+1,t+1} - (1 + r_{t})b_{j,s,t} = w_{t}e_{j,s}n_{j,s,t} + bq_{j,s,t} + tr_{j,s,t} - T_{j,s,t}
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
    Compute the residuals of the household budget constraint at the end of life given a
    guess for labor supply. Solve first for consumption given labor supply and then for
    savings given consumption. Then check the budget constraint.

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
        mtrx_params (list): parameters of the marginal tax rate on labor income functions
        t (int): model period
        j (int): index of ability type
        p (OG-Core Specifications object): model parameters
        method (str): adjusts calculation dimensions based on 'SS' or 'TPI'

        Returns:
            BC_error (array_like): residuals of the household budget constraint
    """
    # change n to array
    n = np.atleast_1d(n)
    # print("EOL n, b = ", n, b)
    # use labor supply equation to get consumption
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
    # check the budget constraint
    # print("EOL c, n, b, bps1 = ", c, n, b, b_splus1)
    BC_error = BC_residual(
        c, n, b, b_splus1, r, w, p_tilde, e, z, bq, net_tax, p
    )
    return BC_error


def HH_system(
    x,
    b_s,
    r_s,
    w_s,
    p_tilde_s,
    factor,
    tr_s,
    ubi,
    bq_s,
    theta,
    e_s,
    z,
    chi_n_s,
    etr_params_s,
    mtrx_params_s,
    j,
    t_s,
    p,
    method,
    r_splus1,
    w_splus1,
    p_tilde_splus1,
    n_splus1_interp,
    c_splus1_interp,
    rho_s,
    etr_params_splus1,
    mtry_params_splus1,
    e_splus1,
    z_index,
):
    r"""
    Computes the residuals for the labor and savings FOCs for a given
    state b_s and guesses for the choice variables n_s and b_splus1.

    Args:
        x (array_like): A vector with guesses for n_s and b_splus1.
        b_s (scalar): Assets in the current period.
        ... (other arguments for the current and next period)
        n_splus1_interp (list): List of interpolation functions for n_{s+1}.
        c_splus1_interp (list): List of interpolation functions for c_{s+1}.

    Returns:
        errors (array_like): A vector of the two FOC errors (labor and savings).
    """
    n_s, b_splus1 = x
    n_s = np.atleast_1d(n_s)

    # Calculate net taxes today given choices
    net_tax_s = tax.net_taxes(
        r_s, w_s, b_s, n_s, bq_s, factor, tr_s, ubi, theta,
        t_s, j, False, method, e_s * z, etr_params_s, p
    )

    # Calculate consumption from the budget constraint
    c_s = get_cons(
        r_s, w_s, p_tilde_s, b_s, b_splus1, n_s, bq_s, net_tax_s, e_s, z, p
    )


    # Labor FOC error
    foc_labor_error = FOC_labor(
        r_s, w_s, p_tilde_s, b_s, c_s, n_s, factor, e_s, z,
        chi_n_s, etr_params_s, mtrx_params_s, t_s, j, p, method
    )

    # Savings FOC (Euler) error
    foc_savings_error = euler_error(
        c_s, b_splus1, r_splus1, w_splus1, p_tilde_splus1, p_tilde_s,
        n_splus1_interp, c_splus1_interp, factor, rho_s,
        etr_params_splus1, mtry_params_splus1, j, t_s + 1 if hasattr(t_s, "__len__") else t_s,
        e_splus1, z_index, p, method
    )

    errors = np.array([foc_labor_error, foc_savings_error])
    return np.squeeze(errors)


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
    """
    Solves the household problem for a given ability type j.
    This function iterates backwards over age s. In each period, it solves
    the household's optimization problem "forward" given assets b_s.
    It returns the policy functions for savings, consumption, and labor supply.
    """
    nb = len(b_grid)

    # Initialize policy function arrays
    b_policy = np.zeros((p.S, nb, p.nz))
    c_policy = np.zeros((p.S, nb, p.nz))
    n_policy = np.zeros((p.S, nb, p.nz))

    # Initialize lists to hold interpolation objects for each age s
    b_policy_interp = [None] * p.S
    c_policy_interp = [None] * p.S
    n_policy_interp = [None] * p.S

    # Solve the last period of life (s = S-1)
    s_EOL = p.S - 1
    current_t_EOL = t[s_EOL] if hasattr(t, "__len__") else t

    for z_index, z in enumerate(p.z_grid):
        for b_index, b in enumerate(b_grid):
            args_EOL = (
                b, p_tilde[s_EOL], r[s_EOL], w[s_EOL], tr[s_EOL], ubi,
                bq[s_EOL], theta, factor, e[s_EOL], z, chi_n[s_EOL],
                etr_params[s_EOL, :], mtrx_params[s_EOL, :],
                current_t_EOL, j, p, method,
            )

            n = opt.brentq(EOL_system, 0.0, p.ltilde, args=args_EOL)
            
            n_policy[s_EOL, b_index, z_index] = n
            c = c_from_n(
                n, b, p_tilde[s_EOL], r[s_EOL], w[s_EOL], factor,
                e[s_EOL], z, chi_n[s_EOL], etr_params[s_EOL, :],
                mtrx_params[s_EOL, :], current_t_EOL, j, p, method
            )
            c_policy[s_EOL, b_index, z_index] = c
            b_policy[s_EOL, b_index, z_index] = b_from_c_EOL(
                c, p_tilde[s_EOL], j, p.sigma, p
            )

    # Create and store interpolation objects for the last period
    c_interp_list, n_interp_list, b_interp_list = [], [], []
    for z_index in range(p.nz):
        c_interp_list.append(itp.PchipInterpolator(b_grid, c_policy[s_EOL, :, z_index], extrapolate=True))
        n_interp_list.append(itp.PchipInterpolator(b_grid, n_policy[s_EOL, :, z_index], extrapolate=True))
        b_interp_list.append(itp.PchipInterpolator(b_grid, b_policy[s_EOL, :, z_index], extrapolate=True))
    c_policy_interp[s_EOL] = c_interp_list
    n_policy_interp[s_EOL] = n_interp_list
    b_policy_interp[s_EOL] = b_interp_list

    # Iterate backwards from s = S-2 to 0
    for s in range(p.S - 2, -1, -1):
        c_splus1_interp = c_policy_interp[s + 1]
        n_splus1_interp = n_policy_interp[s + 1]

        current_t_s = t[s] if hasattr(t, "__len__") else t
        
        for z_index, z in enumerate(p.z_grid):
            for b_index, b_s in enumerate(b_grid):
                args_HH = (
                    b_s, r[s], w[s], p_tilde[s], factor, tr[s], ubi, bq[s],
                    theta, e[s], z, chi_n[s], etr_params[s, :],
                    mtrx_params[s, :], j, current_t_s, p, method,
                    r[s+1], w[s+1], p_tilde[s+1], n_splus1_interp,
                    c_splus1_interp, rho[s], etr_params[s+1, :],
                    mtry_params[s+1, :], e[s+1], z_index
                )
                
                # Make an initial guess for [n_s, b_splus1]
                guess_n = n_policy[s, b_index - 1, z_index]
                guess_b = b_policy[s, b_index - 1, z_index]
                initial_guess = np.array([guess_n, guess_b])


                res = opt.root(HH_system, initial_guess, args=args_HH, method='hybr')
                if not res.success:
                    print(f"Warning: HH_system did not converge for s={s}, b_index={b_index}, z_index={z_index}. Using initial guess.")
                n, b_splus1 = res.x if res.success else initial_guess
                
                n_policy[s, b_index, z_index] = n
                b_policy[s, b_index, z_index] = b_splus1

                net_tax = tax.net_taxes(
                    r[s], w[s], b_s, n_policy[s, b_index, z_index], bq[s], factor,
                    tr[s], ubi, theta, current_t_s, j, False, method, e[s] * z,
                    etr_params[s, :], p
                )
                c_policy[s, b_index, z_index] = get_cons(
                    r[s], w[s], p_tilde[s], b_s, b_policy[s, b_index, z_index],
                    n_policy[s, b_index, z_index], bq[s], net_tax, e[s], z, p
                )

        # Create and store interpolation objects for the current period
        c_interp_list, n_interp_list, b_interp_list = [], [], []
        for z_index in range(p.nz):
            c_interp_list.append(itp.PchipInterpolator(b_grid, c_policy[s, :, z_index], extrapolate=True))
            n_interp_list.append(itp.PchipInterpolator(b_grid, n_policy[s, :, z_index], extrapolate=True))
            b_interp_list.append(itp.PchipInterpolator(b_grid, b_policy[s, :, z_index], extrapolate=True))
        c_policy_interp[s] = c_interp_list
        n_policy_interp[s] = n_interp_list
        b_policy_interp[s] = b_interp_list

    return b_policy, c_policy, n_policy, b_policy_interp, c_policy_interp, n_policy_interp
