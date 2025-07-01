## Part 2 

### Grid search distribution of agents

Here we solve for the stationary distribution of agents across age, productivity and wealth given the choices of households, $h_j(z, a)$. The benefit of grid search is that it guarantees by construction that the policy functions for agents will lie on the assets grid, so the discretization of the steady state distribution follows directly from solving the household problem (and the birth / death rate). First we find the relative size of each cohort of age $j$ (denoted $\mu_j$) with 

$$ \mu_{j+1} = \frac{\mu_j}{1+n} $$ 

for each $j = 1, 2, ..., N-1$.

Then, we normalize so that $\sum_j \mu_j = 1$. Then we get the initial distribution from the intial ergodic distrubution of productivity by $h_1(z_H, 0) = \mu_1 \pi_H$, $h_1(z_L, 0) = \mu_1 \pi_L$. After obtaining the initial distribution from the population growth / death rates and the ergodic distribution of productivity shocks, we can iterate forward with policy functions. We exploit the grid structure of the policy functions, where both the state and the policy function lie on the grid. That is, $a, g(a, z) \in A = \{a_1, a_2, ..., a_K\}$. We use the following equations to iterate forward with policy functions:

$$ \begin{align*}
\text{workers: } \quad h_{j+1}(a', z') &= \sum_{a_k} \sum_{z \in \{z_H, z_L\}} \bold{1}\{a' = g_j(a_k, z)\} \Pi(z, z') h_j(a_k, z) \\

\text{retirees: } \quad h_{j+1}(a') &= \sum_{a_k} \bold{1}\{a' = g_j(a_k)\} h_j(a_k)
\end{align*}
$$

Note that since retired workers are not characterized by their productivity, the distribution of retired workers is just over age and assets. We also must make sure that the distributions are weighted properly across age according to $\sum_{a_k, z} h_j(a_k, z) = \mu_j$.

### EGM steady state distribution

Unfortunately, the endogenous grid method does not share the convenience of grid search for steady state distribution estimation resulting from the already discretized policy functions. Instead, policy functions are continuous and it is unlikely that the policy function will fall exactly on the grid. Thus, when the policy function falls between to grid points, we will follow Young (2010) in splitting the probability across grid points according to their distance. This requries a slightly different process for points that lie outside the grid than for points that lie between grid points. Let the assets grid be $A = \{a_1, a_2, ..., a_K\}$. 

The intial distribution is obtained the same way as the grid search method. However, we iterate forward with different equations as follows:

Workers:
$$ \begin{align*}
\text{Grid min: } \quad h_{j+1}(a_1', z') &= \sum_{a_k} \sum_z \left(\bold{1}\{g_j(a_k, z) \leq a_1' \} + \bold{1}\{ a_1' < g_j(a_k, z) < a_2'\} \frac{a_2' - g_j(a_k, z)}{a_2' - a_1'}\right) \Pi(z, z') h_j(a_k) \\

\text{Interior point: } \quad h_{j+1}(a_\kappa', z') &= \sum_{a_k} \sum_z \left(\bold{1}\{a_{\kappa -1}'  < g_j(a_k, z) \leq a_\kappa' \}\frac{g_j(a_k, z) - a_{\kappa - 1}'}{a_\kappa' - a_{\kappa - 1}'} + \bold{1}\{ a_\kappa' < g_j(a_k, z) < a_{\kappa + 1}'\} \frac{a_{\kappa + 1}' - g_j(a_k, z)}{a_{\kappa + 1}' - a_\kappa'}\right) \Pi(z, z') h_j(a_k)\\

\text{Grid max: } \quad h_{j+1}(a_K', z') &= \sum_{a_k} \sum_z \left(\bold{1}\{ a_{K-1} < g_j(a_k, z) < a_K\} \frac{g_j(a_k, z) - a_{K-1}'}{a_K' - a_{K-1}'} + \bold{1}\{g_j(a_k, z) \geq a_K' \}\right) \Pi(z, z') h_j(a_k)
\end{align*}
$$

Retirees:
$$ \begin{align*}
\text{Grid min: } \quad h_{j+1}(a_1') &= \sum_{a_k} \left(\bold{1}\{g_j(a_k) \leq a_1' \} + \bold{1}\{ a_1' < g_j(a_k) < a_2'\} \frac{a_2' - g_j(a_k)}{a_2' - a_1'}\right) h_j(a_k) \\

\text{Interior point: } \quad h_{j+1}(a_\kappa') &= \sum_{a_k} \left(\bold{1}\{a_{\kappa -1}'  < g_j(a_k) \leq a_\kappa' \}\frac{g_j(a_k) - a_{\kappa - 1}'}{a_\kappa' - a_{\kappa - 1}'} + \bold{1}\{ a_\kappa' < g_j(a_k) < a_{\kappa + 1}'\} \frac{a_{\kappa + 1}' - g_j(a_k)}{a_{\kappa + 1}' - a_\kappa'}\right) h_j(a_k)\\

\text{Grid max: } \quad h_{j+1}(a_K') &= \sum_{a_k} \left(\bold{1}\{ a_{K-1} < g_j(a_k) < a_K\} \frac{g_j(a_k) - a_{K-1}'}{a_K' - a_{K-1}'} + \bold{1}\{g_j(a_k) \geq a_K' \}\right) h_j(a_k)
\end{align*}
$$


---

### Stochastic simulation:

The last two methods for estimating the distribution of agents exploited the asset grid and used policy functions to numerically approximate the distribution of agents. However, a more intuitive and realistic method for estimating the distribution of agents is to do so with stochastic simulation. Here is a sketch of the algorithm:

1. Simulate $M$ (large, say 10,000) draws from the Markov process across the lifetime, with zero assets at birth. Each draw is a vector length $J_r -1$ that represents the productivity shocks that a household faces throughout their lifetime.
2. Use the household asset policy functions to get the household decisions at each time period. Keep track of the asset holdings at each time period.
3. We should have a panel data set of $M$ households and their age across the $N$ periods of their lifetime, with productivity (while working) and asset holdings at each age. 
4. Bin the data within each age and productivity group according to the asset grid, $A = \{a_1, a_2, ..., a_K\}$. Weight each age group $j$ according to $\mu_j$.
5. Should have an approximate density function $h_j(a, z)$.

Computationally, it makes more sense to have a separate distribution for working and retired agents, as they have different state spaces and retired households are simpler. We can collapse this later to see the population distribution of assets across all age groups. 





---
# Seshadri and Lee

Certainly. Here's a summary of the model and solution method from the Seshadri and Lee paper:

Model Summary

The model is an overlapping generations framework with 13 discrete periods, each representing 6 years. The key components are:

1. Human Capital Accumulation:
   Adults accumulate human capital following a Ben-Porath technology:

   $h_{j+1} = e_{j+1}[a(n_j h_j)^\beta + h_j]$

   where $a$ is learning ability, $n_j$ is time spent on human capital accumulation, and $e_{j+1}$ is a market luck shock.

2. Childhood Skill Formation:
   Children's human capital is formed over three periods (ages 0-17) according to:

   $h'_3 = z[(q_2 X_2^{\phi_2} + (1-q_2)(q_1 X_1^{\phi_1} + (1-q_1)q_0 X_0^{\phi_0})^{\frac{\phi_1}{\phi_0}})^{\frac{\phi_2}{\phi_1}}]^{\frac{1}{\phi_2}}$

   where $X_j$ is a composite of parental time and goods investments.

3. College Decision:
   At age 18, individuals decide whether to attend college based on costs and expected returns.

4. Intergenerational Transfers:
   Parents make financial transfers to children when children become adults.

5. Government Policies:
   The model includes progressive taxation, education subsidies, and social security.

Solution Method

The model is solved using the following steps:

1. Value Function Iteration:
   The authors use backward induction to solve for value functions and policy functions at each life stage.

2. Equilibrium Computation:
   They solve for equilibrium prices (interest rate and wages) that clear markets.

3. Simulated Method of Moments:
   Model parameters are estimated by minimizing the distance between simulated moments and empirical moments:

   $\hat{\Theta} = \arg\min_{\Theta} [M(\Theta) - M_s]' [M(\Theta) - M_s]$

   where $M(\Theta)$ are simulated moments and $M_s$ are empirical moments.

4. Numerical Implementation:
   - They use a grid for continuous state variables and interpolate value functions.
   - The AR(1) process for abilities is approximated using the Rouwenhorst method.
   - Optimization is performed using a projected quasi-Newton method with boundary constraints.

5. Stationary Distribution:
   They obtain the stationary distribution through Monte Carlo simulation, simulating 120,000 households for 200 generations.

6. Counterfactuals:
   Policy experiments are conducted by changing relevant parameters and recomputing the equilibrium.

The model is computationally intensive due to its many state variables and choice variables. The authors make several simplifications to make the problem tractable, such as using value function iteration instead of policy function iteration, and simulating the distribution rather than directly approximating it.

---

Model Structure

The model is an overlapping generations framework with 13 discrete periods, each representing 6 years. Individuals live from age 0 to 78, with childhood lasting until age 23.

Household Problem

The household problem is structured differently for various life stages:

1. Young Adulthood (Period j = 4, age 24-29):

Individuals solve a standard life-cycle savings problem:

$V_4(S,a,h_4,s_4) = \max_{c_4,s_5,n_4} \{u(c_4) + \beta \int V_5(a';S,a,h_5,s_5)dF(e_5)dG(a'|a)\}$

subject to:
$c_4 + s_5 = f_4(e_4,0) + s_4$
$e_4 = w_S h_4(1-n_4)$
$n_4 \in [0,1]$
$s_5 \geq -g/(1+r)$

where $S$ is education status, $a$ is ability, $h_4$ is human capital, $s_4$ is initial assets, $c_4$ is consumption, $n_4$ is time spent on human capital accumulation, and $f_4(\cdot)$ is after-tax income.

2. Parenting Stage (Periods j = 5,6,7, ages 30-47):

Parents make decisions for themselves and invest in their children:

$V_j(a',\tilde{h}'_{j-4};S,a,h_j,s_j) = \max_{C_j,s_{j+1},n_j,l_{j-5},m_{j-5}} \{qu(C_j) + \beta \int V_{j+1}(a',\tilde{h}'_{j-3};S,a,h_{j+1},s_{j+1})dF(e_{j+1})\}$

subject to:
$C_j + s_{j+1} = f_j(e_j,s_j) + s_j$
$e_j = w_S h_j(1-n_j-l_{j-5}) - m_{j-5}$
$n_j \in [0,1], l_{j-5} \in [0,n_j]$
$s_{j+1} \geq -g/(1+r)$

where $l_{j-5}$ and $m_{j-5}$ are time and goods investments in children, respectively.

3. College Decision (Period j = 8, age 48-53):

Parents and children jointly decide on college attendance:

$V_8(a',h'_3;S,a,h_8,s_8) = \max_{S'} \{W_8(S',a',h'_3;S,a,h_8,s_8) + \omega_S \cdot S'\}$

where $S'$ is the college decision and $\omega_S$ is a preference for college.

4. Inter-vivos Transfers (Period j = 9, age 54-59):

Parents make financial transfers to their now-adult children:

$V_9(S',a',h'_4;S,a,h_9,s_9) = \max_{c_9,s_{10},n_9,s'_4} \{u(c_9) + vV_4(S',a',h'_4,s'_4) + \beta \int V_{10}(S,a,h_{10})dF(e_{10})\}$

subject to:
$c_9 + s_{10} + s'_4 = f(e_9,s_9) + s_9$
$e_9 = w_S h_9(1-n_9)$
$s'_4 \geq 0$

5. Retirement (Periods j = 10,11,12, ages 60-77):

Retirees live off social security benefits and savings:

$V_{10}(S,h_{10},s_{10}) = \max_{c_{10},s_{11}} \{u(c_{10}) + \beta u(c_{11}) + \beta^2 u(c_{12})\}$

subject to:
$\sum_{j=10}^{12} c_j(1+\tilde{r})^{j-10} = f(e_{10},s_{10}) + s_{10} + \frac{2+\tilde{r}}{(1+\tilde{r})^2}(p_0 + p_1 e_{10} + g)$

where $p_0$ and $p_1$ capture the social security scheme.

Key Constraints and Technologies

1. Human Capital Accumulation:
   $h_{j+1} = e_{j+1}[a(n_j h_j)^\beta + h_j]$

2. Childhood Skill Formation:
   $h'_3 = z[(q_2 X_2^{\phi_2} + (1-q_2)(q_1 X_1^{\phi_1} + (1-q_1)q_0 X_0^{\phi_0})^{\frac{\phi_1}{\phi_0}})^{\frac{\phi_2}{\phi_1}}]^{\frac{1}{\phi_2}}$

   where $X_j = (l_j h_j + g_j d_j/w_S)^{\gamma_j}[m_j + (1-\gamma_j)d_j]^{1-\gamma_j}$

3. Intergenerational Transmission of Ability:
   $\log a' = (1-\rho_a)(\mu_a - \sigma_a^2/2) + \rho_a \log a + \eta'$, $\eta \sim N(0,(1-\rho_a^2)\sigma_a^2)$

4. Government Budget Constraint:
   $2(p_0 + p_1 \int e_{10}(z_{10})dF(z)) = 8\tau_s \bar{e}$

This structure captures the complex interactions between human capital investment, intergenerational transfers, and life-cycle decisions, subject to various constraints including borrowing limits and government policies.