# Household problem with idiosyncratic income shocks

This note documents the household model and the solver in
`household_stoch.py` and `household_stoch_kernels.py`. It reconciles
the derivation in `OG-Core/ogcore/hh.md` (untracked notes on the 2024
`stoch_inc` branch) with OG-Core 0.20's household code and replaces it
for this project; the theory section of the eventual OG-Core docs
(Phase 6) should start from here.

## What changed from `hh.md`

| `hh.md` | Here (and in the code) | Why |
| --- | --- | --- |
| Bellman equation puts $\beta_j$ on the bequest term | No $\beta_j$ on the bequest term | The bequest is chosen and valued at age $s$; OG-Core's `FOC_savings` (and `c_from_b_splus1`) have no $\beta_j$ there |
| Labor FOC has $e^{g_y(1-\sigma)}$ on $\chi^n_s h'(n)$ | No growth factor | OG-Core's detrended `FOC_labor` has none; the factor is absorbed when the utility function is made stationary |
| Savings Euler equation without $e^{-\sigma g_y}$ | $e^{-\sigma g_y}$ on the right-hand side | Detrending; matches `FOC_savings` (the final formula in `hh.md` had it) |
| Budget: $b_{s+1}$, labor income $w z n$, no remittances, pensions or minimum consumption | $e^{g_y} b'$, labor income $w e z n$, remittances, pensions, $c_{min}$, consumption taxes via $\tilde p$ | OG-Core's `get_cons` |
| Bequests $\zeta_{j,s} BQ / (\lambda_j \omega_s)$ | $BQ_j / \sum_s \omega_{s,j}$ (`use_zeta = False`) | `get_bq` since OG-Core PR #1165 |
| EGM step solves for $(b, n)$; policies interpolated separately in $c$, $n$, $b'$ | Savings $b'$ interpolated; $n$ and $c$ re-solved exactly at every grid node | Budget constraint holds exactly at every node (needed for the resource constraint in GE) |
| No borrowing-constraint treatment | Explicit constrained region at $b_{min} = 0$; Euler equation checked at every node and solved directly where interpolation misses | Phase 2 |
| $0 < n < \tilde l$ | Labor may sit at a bound (corner), where the labor FOC is an inequality | OG-Core's analytical MTRs exceed 100% at low labor income for households with high capital income |

## Model

Households of lifetime-income type $j$ live for $S$ model periods
(ages $s = 0, \dots, S-1$, OG-Core's $E+1, \dots, E+S$). Labor
productivity is $e_{j,s} z$: OG-Core's deterministic profile times an
idiosyncratic shock $z$ that follows a Markov chain. All variables are
detrended by productivity growth $e^{g_y t}$.

**Preferences.** Period utility $u(c) = (c^{1-\sigma} - 1)/(1-\sigma)$,
disutility of labor $\chi^n_s h(n)$ with OG-Core's elliptical $h$
(`marg_ut_labor`, stitched linearly within $10^{-6}$ of $0$ and
$\tilde l$), and a warm-glow bequest motive
$\chi^b_j b'^{1-\sigma}/(1-\sigma)$. With mortality $\rho_s$ after age
$s$ ($\rho_{S-1} = 1$),

$$
V_{j,s}(b, z) = \max_{c, n, b'}\;
u(c) - \chi^n_s h(n)
+ e^{(1-\sigma) g_y}\Bigl[\rho_s \chi^b_j \tfrac{b'^{1-\sigma}}{1-\sigma}
+ \beta_j (1 - \rho_s)\, \mathbb{E}\bigl[V_{j,s+1}(b', z') \mid z\bigr]\Bigr]
$$

subject to the budget constraint and $b' \ge b_{min} = 0$,
$\underline n \le n \le \tilde l$. The borrowing limit is zero: OG-Core's
tax functions are not defined for negative capital income, and asset
grids must start at 0. The lower bound on labor $\underline n$ is 0, or,
when $\upsilon > 2$, just above the point where OG-Core's linear
stitching of $h'$ below $n = 10^{-6}$ turns negative (`labor_floor`;
about $5 \times 10^{-7}$ in the default calibration).

**Budget constraint** (OG-Core's `get_cons` with productivity $e z$):

$$
\tilde p\, c + e^{g_y} b' = (1 + r) b + w e_{j,s} z n + bq_{j,s} + rm_{j,s}
+ tr_{j,s} + ubi_{j,s} + pension_{j,s}
- T^I(b, n) - T^W(b) - \tau^{bq} bq_{j,s}
- \textstyle\sum_i (1 + \tau^c_i) p_i c_{min,i}
$$

with $X = w e z n \cdot factor$ and $Y = r b \cdot factor$ (dollars),

$$
T^I = f^I \,\tau^{etr}(X, Y)\,\frac{X(1 - \nu^l) + Y(1 - \nu^k)}{factor}
+ \tau^p w e z n,
\qquad
T^W = f^W \frac{p^w h^w b^2}{h^w b + m^w},
$$

where $f^I, f^W$ are the income- and wealth-tax filer shares of type
$j$ and $\nu^l, \nu^k$ the noncompliance rates. This is
`ogcore.tax.income_tax_liab` (the `ETR_income` noncompliance adjustment
$(X\nu^l + Y\nu^k)/(X+Y)$ written out) plus `wealth_tax_liab`.

**First-order conditions.** Labor (with $\le$ at $n = \underline n$ and
$\ge$ at $n = \tilde l$):

$$
\frac{u'(c)}{\tilde p}\, w e_{j,s} z \bigl(1 - \tau^p - f^I(1-\nu^l)\tau^{mtrx}(X, Y)\bigr)
= \chi^n_s h'(n).
$$

Savings, for $s < S - 1$ (an inequality, $\ge$, when $b' = b_{min}$):

$$
\frac{u'(c)}{\tilde p} = e^{-\sigma g_y}\Bigl[\rho_s \chi^b_j b'^{-\sigma}
+ \beta_j (1-\rho_s)\,\mathbb{E}\Bigl[\frac{1 + r' - r' f^I(1-\nu^k)\tau^{mtry}(X', Y') - \tau^{mtrw}(b')}{\tilde p'}\, u'(c')\Bigm| z\Bigr]\Bigr].
$$

At the last age, $u'(c)/\tilde p = e^{-\sigma g_y}\rho_{S-1}\chi^b_j b'^{-\sigma}$,
so $b' = \kappa c$ with $\kappa = e^{-g_y}(\tilde p \rho_{S-1}\chi^b_j)^{1/\sigma}$.

$\tau^{mtrw}$ is OG-Core's `MTR_wealth`, which multiplies the first term
of the derivative of $p^w h^w b^2/(h^w b + m^w)$ by the filer share
twice. For $0 < f^W < 1$ it is therefore not exactly $\partial T^W /
\partial b$; the code follows OG-Core.

The code uses OG-Core's marginal utility of consumption (stitched
linearly below $c = 0.003$), its exact inverse, and a bequest marginal
utility stitched linearly below $b' = 0.001$ so that it is finite at
the borrowing limit. Marginal tax rates come from the separately
estimated MTR functions unless `analytical_mtrs` is set, exactly as in
OG-Core, so the FOCs are those of `FOC_labor` and `FOC_savings`.

**Shocks.** $\log z' = \rho_z \log z + \varepsilon$,
$\varepsilon \sim N(0, \sigma_z^2)$, discretized by Rouwenhorst's method
(`rouwenhorst`, `set_ar1_shock_process`): the chain matches the AR(1)'s
autocorrelation and unconditional variance exactly; levels are scaled so
that $\mathbb{E}[z] = 1$ under the stationary distribution, which keeps
the meaning of $e_{j,s}$. Newborns draw $z$ from the stationary
distribution unless `z_dist0` is given. `set_shock_process` takes any
grid and transition matrix directly. One state ($z = 1$) is OG-Core.

**Pensions.** Benefits are computed per $(s, j)$ outside the
household's grid (`pension_ss`) and enter the budget as income the
household takes as given, as in OG-Core, whose FOCs also ignore the
effect of labor on benefits. With shocks they come from each type's mean
effective labor profile $\mathbb{E}[z n \mid s, j]$: exact with one
state, and a documented simplification otherwise (no earnings-history
state). For US-style Social Security, $\theta_j$ comes from
`ogcore.pensions.replacement_rate_vals` on that profile; Defined
Benefits, Notional Defined Contribution and Points System use
`pension_amount`. `solve_all_households_ss` without `theta` finds the
fixed point, as OG-Core's household solve does. (OG-Core's `SS_amount`
places US-style benefits by array position, so evaluating it at a
single age gives zero; this is why Phase 1 had no pensions.)

## Solution method (`solve_lifecycle`)

All types in a batch are solved together, age by age backward; every
step is vectorized over (type, asset node, shock), with numba kernels
for the per-point work.

**Within-period problem** (`_static_solve`). Given assets $b$ and a
savings rule $b' = b'_0 + \kappa c$, consumption comes from the budget
constraint at each $n$, so the budget holds exactly whatever $n$ is,
and $n$ solves the labor FOC $F(n) = u'(c(n))\, w e z D(n) / \tilde p -
\chi^n_s h'(n) = 0$, $D = 1 - \tau^p - f^I (1 - \nu^l)\tau^{mtrx}$, by
safeguarded Newton with bisection on $[\underline n, \tilde l]$ (to
$10^{-12}$ relative to the two terms). If $F(\underline n) \le 0$ (for
example where $D < 0$) or $F(\tilde l) \ge 0$, labor supply is at that
bound (a corner); the bounds are only evaluated when Newton steps head
there. With analytical MTRs, $F$ need not be monotone (the MTR exceeds
100% at low labor income for households with high capital income), so
not working and an interior root can both be local optima: the solver
then also checks the corner at $\underline n$, scans $F$ at 12 points
for roots where it turns from $+$ to $-$, and keeps the candidate with
the highest within-period utility ($u(c)$, weighted by
$1 + \kappa e^{g_y}/\tilde p$ at the last age, where the bequest moves
with $c$, plus the utility of leisure). With estimated MTRs OG-Core's
household is defined by its FOCs, which need not be those of any
utility comparison (at extreme incomes the ETR can make after-tax
income fall with labor while the MTR function says otherwise), so the
root the search reaches is the solution.

1. **Last age.** Solve the within-period problem with $b' = \kappa c$.
2. **Euler step** (`_value_of_savings`). For each savings level $b'$ on
   the grid and current shock, the savings Euler equation gives the
   marginal utility of consumption $v$ and $c^E(b') = (u')^{-1}(v)$,
   using next age's policies at the grid nodes (no interpolation). Knots
   with $v$ above $u'(0)$ would need $c \le 0$; they are never optimal
   ("infeasible knots": small $b'$ where the bequest motive is strong;
   7% of knots in the default five-state calibration, at ages 25-78 and
   $b' \lesssim 0.01$). A non-finite $v$ counts as an inversion failure.
3. **Inversion.** For each feasible knot $(c^E, b')$, find the assets
   $b$ and labor $n$ at which it is optimal: the labor FOC and budget
   constraint, two equations in $(b, n)$ (taxes depend on both). Plain
   Newton with analytic derivatives of the tax functions, started from a
   linear extrapolation of the previous two knots, converges in about
   three iterations; otherwise a bracketed method takes over: solve the
   labor FOC for $n$ at given $b$ (a corner if its sign never changes),
   then find the root of the budget residual $H(b)$ by safeguarded
   Newton. If $H(0) \ge 0$, the household at the borrowing limit can
   afford more than $(c, b')$, so the knot lies below the limit; there
   taxes are those at $b = 0$ and $H$ is linear, which places the knot.
   A bracket that collapses where $H$ jumps (no root) is a failure, and
   the knot is dropped.
4. **Savings at grid nodes** (`_to_exogenous`). Knots are kept where
   they increase in $b$ (scanning down from the top, so a knot out of
   order low down removes the knots below it), and $b'(b)$ is
   interpolated with a monotone cubic (scipy's PCHIP slopes) through the
   knots at or above the borrowing limit only: the extension below zero
   assets has a kink at $b = 0$. If the lowest of those knots is the one
   for $b' = 0$, nodes below it are in the **constrained region**,
   $b' = 0$. Otherwise the knots for low $b'$ lie below the limit, and
   the nodes between the limit and the first knot above it ("gap"
   nodes) get $b'$ from the Euler equation directly
   (`_solve_euler_direct`): the root of $\phi(b') = c(b, b') - c^E(b')$,
   with $c(b, b')$ from the within-period problem and $c^E$ interpolated
   (PCHIP) over the knots in $b'$, where it is smooth; if
   $c(b, 0) \le c^E(0)$ the constraint binds. The root is bracketed by
   stepping outward over grid nodes from the guess (up to twice the top
   of the grid) and found by regula falsi (Illinois) to $10^{-11}$ in
   consumption. A type and shock with fewer than two usable knots at or
   above the limit (the inversion failed for the rest, as at savings far
   above anyone's when the first grid is too large) is solved directly
   at every node.
5. **Labor and consumption at grid nodes.** Given $b'$, solve the
   within-period problem. The budget holds to machine precision at
   every node.
6. **Euler check at every node.** The relative error $c^E(b')/c - 1$
   (at $b' = 0$, only a violated inequality, $c > c^E(0)$, counts) is
   computed at every node; where it exceeds `SolverOptions.euler_tol`
   ($10^{-5}$; at least $10^{-9}$), $b'$ is solved for directly as in
   step 4. This repairs nodes where interpolating knots in $b$ is
   inaccurate or wrong: where the tax functions make the budget
   non-monotone in $b$ near zero assets (so the policy is not monotone
   and knots are dropped), next to kinks, and where labor switches
   between a corner and an interior choice. In the default five-state
   calibration 28% of nodes are solved directly (19% are gap nodes),
   which takes about 40% of the solve time; `euler_tol = 1e-4` is faster
   and less accurate. Each node's search runs in a compiled kernel
   (`_euler_direct_point`). With analytical MTRs, $\phi$ can jump from
   $+$ to $-$ where the labor choice switches; the optimum is then at
   the jump, a kink of the value function where the Euler equation
   holds as a pair of inequalities (`euler_kinks`). Otherwise a jump is
   a failure.

**Failures** are loud: every solve reports its status;
`SolverDiagnostics` counts within-period failures (including $c \le 0$),
inversion failures and nodes whose Euler error stays above `euler_tol`,
together with constrained, corner, gap, extrapolated, directly solved
and kink nodes and dropped or infeasible knots, and marks the failed
nodes (`failed_nodes`); it logs a warning with the ages affected. A
failed within-period solve at a node is retried through the direct
Euler solve before it counts. `solve_all_households_ss` weighs failed
nodes by the population there (`failed_mass`) and raises
`HouseholdSolverError` when that share exceeds
`SolverOptions.max_fail_frac` ($10^{-4}$), so failures at nodes no
household reaches (for example where OG-Core's tax functions misbehave
at extreme incomes) are logged but do not stop the solve;
`solve_lifecycle` on its own, without a distribution, compares the
number of failed nodes with the same fraction. Both raise when more
than that fraction of knots fail or a policy is NaN (naming the age).
Inputs are checked first: NaN or infinite prices, receipts, tax
parameters or grids, a $z$ chain whose rows do not sum to 1, or a grid
not starting at 0 raise `ValueError`.

**Asset grids** are per type and start at the borrowing limit 0:
`make_asset_grid` places $b = b_{max} x^{4}$ for $x$ evenly spaced on
$[0, 1]$ (100 nodes by default). `solve_all_households_ss` resizes
$b_{max}$ until it is within a factor of 1.25 of 1.5 times the highest
savings chosen where the distribution has mass (more than $10^{-10}$ of
the type's population; failed nodes left out), for at most four rounds,
and logs a warning if the grids have not settled
(`iterations["grid_settled"]`). Type 6 of the default calibration (1%
of the population, per-capita bequests of 2) saves far more than the
rest; Phase 1's common grid stopped below its savings.

## Distribution (`get_ss_distribution`)

Young's (2010) method: the mass at node $(b_i, z)$ at age $s$ moves to
the two nodes around $b'(b_i, z)$ with weights that preserve the mean,
and across shocks by the transition matrix. Mortality and immigration do
not depend on assets or shocks, so the within-cell distribution evolves
by the policies alone; each $(s, j)$ cell is then scaled to
`omega_SS[s, j]`, so the distribution sums to the population shares and
to one overall. Newborns hold $b = 0$, a grid node. Mass that would leave
the grid is clipped and reported (`mass_above_grid`, `mass_below_grid`),
as is the mass at the top node. A NaN or infinite savings policy where
the distribution has mass raises `ValueError`.

## Backends

`SolverOptions(backend="auto")` uses the numba kernels when the tax
functions are parametric (DEP, DEP_totalinc, GS, HSV, linear) and
otherwise the numpy backend, which solves the same equations on
OG-Core's own `get_tax_rates` (evaluated at a minimum income of
$10^{-6}$ dollars for GS and HSV, which are singular at zero income, as
in the kernels) with finite-difference Jacobians, so nonparametric
"mono" taxes work, 10-20 times more slowly. Its within-period and
direct Euler solves follow the kernels step by step; its inversion
(step 3) has only the bracketed method, without the plain Newton stage
and the continuation from one knot to the next, so where the inversion
has several roots it can find another one (nodes are checked against
the Euler equation either way). Set `OGCORE_STOCH_DISABLE_NUMBA=1` to
force numpy kernels. The per-point loops run in parallel on all cores;
limit them with `NUMBA_NUM_THREADS` (e.g. when OG-Core runs Dask
workers).

## Accuracy and speed (Gate 2)

Fixed prices $r = 0.05$, $w = 1.2$, $BQ_j = 0.02$, $TR = 0.01$,
$factor = 100{,}000$, default calibration (J = 7, S = 80).

| Check | Result |
| --- | --- |
| One state vs OG-Core `SS.inner_loop`, all 7 types, pensions off | max gap $3.2 \times 10^{-5}$ in $b'$, $3.0 \times 10^{-6}$ in $n$ (tolerance $10^{-4}$) |
| Same with pensions and transfers on ($\theta$ from OG-Core) | $3.0 \times 10^{-5}$ in $b'$, $4.0 \times 10^{-6}$ in $n$; own fixed-point $\theta$ within $10^{-4}$ of OG-Core's |
| Every budget term on (c_min, consumption tax, remittances, UBI, wealth/bequest/payroll taxes, a non-filer type, noncompliance) | $2.5 \times 10^{-5}$ in $b'$, $3.3 \times 10^{-6}$ in $n$ |
| Grid refinement (nb = 50, 100, 200) | gap $1.8 \times 10^{-4}$, $3.2 \times 10^{-5}$, $3.1 \times 10^{-6}$ |
| OG-Core's own FOCs along the solver's one-state profiles: analytical MTRs, GS, HSV, DEP_totalinc (estimated and analytical MTRs), $\tilde l$ = 0.8 and 1.3, fractional filers with noncompliance and a wealth tax, numpy backend | residuals at most $5 \times 10^{-5}$ of marginal utility (types 0, 3, 6) |
| Five states ($\rho_z = 0.9$, $\sigma_z = 0.15$): Euler errors between nodes where households are | max $1.7 \times 10^{-5}$ (target $10^{-4}$); $3.4 \times 10^{-4}$ anywhere (ages 34-63 near zero assets, no households) |
| Five states: failures, extrapolated nodes, mass at the grid top | none |
| Five states, no bequest motive or transfers (the constraint binds for 84% of households) | no failures; Euler inequality at constrained nodes; mean Euler error $3.6 \times 10^{-6}$, 0.08% of households above $10^{-3}$ (next to kinks) |
| Five states, GS taxes; HSV taxes with analytical MTRs | no failures; Euler errors where households are at most $1.7 \times 10^{-5}$ and $1.6 \times 10^{-5}$ |
| Five states, analytical MTRs | no failures (one dropped knot); 14,833 labor corners; at every node with households, no labor supply on a fine grid gives higher within-period utility; mean Euler error $3.5 \times 10^{-6}$, 0.017% of households above $10^{-3}$ (next to switches between working and not) |
| SS household solve, J = 7, five states, nb = 100 | 1.0 s on 2 cores, 1.7 s on one |

Robustness, five states unless noted (failures; largest Euler error
between nodes where households are, and the mass-weighted mean):

| Variant | Result |
| --- | --- |
| DEP_totalinc taxes; fractional filers, noncompliance and a wealth tax; $\tilde l$ = 0.8 or 1.3; $\sigma$ = 1 or 3; $\chi^b \times 10$; $\upsilon$ = 1.2, 1.5, 2 or 4 | none; $1.6$-$2.7 \times 10^{-5}$ |
| $\sigma_z = 0.4$, $\rho_z = 0.97$; seven states; both | none; $2.6$, $1.9$, $4.0 \times 10^{-5}$ |
| Coarser grids: nb = 30; nb = 20; curvature 2 | none; $7.9 \times 10^{-4}$, $6.0 \times 10^{-3}$, $1.1 \times 10^{-4}$ |
| nb = 30 with curvature 2 and $\sigma_z = 0.4$ | none, but errors up to 0.19 and 30% of households above $10^{-3}$: the first nodes above zero (0.04 to 0.4) are too far apart where young households are and the DEP tax rates change fastest. Keep the defaults (nb = 100, curvature 4) or check `euler_errors` |
| $\sigma_z = 0.4$, $\rho_z = 0.97$ with analytical MTRs; without a bequest motive | none (one dropped knot); max 0.11 and 0.037, mean $1.3 \times 10^{-5}$ and $9.7 \times 10^{-6}$, 0.10% and 0.18% of households above $10^{-3}$ |
| $\sigma = 1$ with $\chi^b \times 10$ | fails (raises) for $10^{-4}$ of the population: mostly type-6 households at ages 47-78 holding about 300, where OG-Core's DEP ETR implies a marginal tax on labor above 100% (2.5 at $n = 0.24$) while its MTR function says 0.48; the labor FOC then jumps to working the whole endowment with $c < 0$, and the Euler equation has no root |
| $\chi^b \times 100$ | 32 Euler failures at nodes holding $1.5 \times 10^{-8}$ of the population (logged, not raised) |

Precautionary saving at fixed prices: aggregate savings 2.526 with one
state, 2.560 with $\sigma_z = 0.1$, 2.663 with $\sigma_z = 0.2$.
Continuity: with $\sigma_z = 10^{-3}$ the largest gap between the
five-state and one-state cell means ($2.8 \times 10^{-5}$ in $b'$) is
$10^{-4}$ of the gap at $\sigma_z = 0.1$ (0.24), the $\sigma_z^2$ scaling
of a second-order effect.

A transition-path sweep (320 periods × 80 ages) is 320 steady-state
solves' worth of age blocks: roughly 5 minutes on 2 cores and 9 on one
at these timings (an estimate; Phase 4 builds it), less with more cores
or `euler_tol = 1e-4`, against the plan's 8-minute target.

## Things to know

* **DEP taxes near zero capital income.** At ages 34-63 OG-Core's
  default DEP parameters raise the tax rate on *all* income steeply with
  the first dollars of capital income (the capital-income term of the
  ETR jumps from its minimum within a few hundred dollars). After-tax
  resources then fall with assets for $b$ just above zero, so savings
  and consumption can fall with assets there, and the EGM knots are not
  monotone; the direct Euler solves handle it. No households are there
  in the default calibration (the tests check monotonicity wherever
  there is mass). Without a bequest motive many are: households with no
  assets save while those with a little are constrained, so the
  constrained region is not the bottom of the grid at those ages.
  OG-Core never evaluates its household problem there.
* **Kinks and off-grid Euler errors.** Where the borrowing constraint
  starts to bind, at this age or the next, policies have kinks that
  interpolation between nodes does not resolve, so Euler errors between
  nodes next to them are larger (1% in the five-state case without a
  bequest motive, 3.7% with $\sigma_z = 0.4$), though few households
  are there. Grid-node values, which are all that Young's method uses,
  satisfy the Euler equation against the interpolated $c^E$; $c^E$
  itself is less accurate across next period's kinks. The mean and the
  population shares that `euler_errors` reports show how much this
  matters; its maximum "where households are" counts any mass at all.
* **Analytical MTRs** (`analytical_mtrs = True`) exceed 100% at low
  labor income for households with high capital income, so those
  households do not work, and the within-period problem can have a
  corner and an interior local optimum. The solver keeps the better one
  (see above), so labor can jump between neighboring nodes, and Euler
  errors between such nodes are large (up to 5% in the default
  five-state case, 11% with $\sigma_z = 0.4$), at nodes holding little
  of the population. With one state the solution still matches
  OG-Core's (gaps below $3.2 \times 10^{-5}$ in $b'$).
* **Extreme calibrations.** The stitched marginal utility of
  consumption is finite at $c = 0$, so a strong enough bequest motive
  can make the Euler equation call for $c < 0$ (infeasible knots), and
  at extreme incomes OG-Core's estimated MTR and DEP ETR disagree about
  whether working more raises income. Where such points carry mass the
  solver reports failures and raises (see the robustness table); where
  they do not, it logs them.
* **Distribution smearing.** Young's method spreads each household over
  two nodes, so aggregates from the distribution differ from the exact
  one-state profiles at $O(h^2)$: with one state and 100 nodes,
  $\sum \omega E[b']$ is 0.04% above OG-Core's and effective labor 0.02%
  above. This matters for Gate 3's 0.1% tolerances; finer grids or a
  lottery on the policies' own values can reduce it.
* **Monotonicity in $z$** is not guaranteed: rich type-6 households near
  retirement work a little more and save slightly less when $z$ is
  higher (the DEP functions couple labor and capital income).
* **Within-cohort consumption dispersion** with $\rho_z = 0.9$ and
  newborns drawn from the stationary distribution rises over working
  life for middle types but is flat for the lowest type (no assets to
  smooth with at labor-market entry, buffers later). Phase 5's
  "dispersion widens with age" check depends on the calibration.
