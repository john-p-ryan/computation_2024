#%%
from OLG_ss_stripped import nonuniform_grid, OLGModel, solve_HH_SS, SS_distribution, solve_SS
import numpy as np
import matplotlib.pyplot as plt
import time



#%%
# read in ef.txt as a numpy array
ef = np.loadtxt('ef.txt', delimiter=',')
eta_end = np.linspace(ef[-1], .5, 67-len(ef))
eta = np.concatenate([ef, eta_end[1:]])

# plot ef across [1, .., len(ef)]
x = np.arange(1, len(eta)+1)
plt.plot(x, eta)
plt.xlabel('Age')
plt.ylabel('Labor efficiency')
plt.title('Age-efficiency profile')


#%%
b_grid = nonuniform_grid(0.0, 22.0, 200, 2.0)
og = OLGModel(b_grid=b_grid, eta=eta, z_grid=np.array([1.0, .25]), gamma=2.5)


#%%
begin = time.time()
b_policy, c_policy, n_policy = solve_HH_SS(og, r=.05, w=1.2)
end = time.time()
print(f'Time elapsed: {end-begin} seconds')


#%%
j = 10

# plot b policy at age j
plt.plot(b_grid, b_policy[j-1, :, 0], label=f'z={og.z_grid[0]}')
plt.plot(b_grid, b_policy[j-1, :, 1], label=f'z={og.z_grid[1]}')
plt.xlabel('b')
plt.ylabel('b\'')
plt.title(f'b policy at age {j}')
plt.legend()

#%%
# plot n policy at age j
plt.plot(b_grid, n_policy[j-1, :, 0], label=f'z={og.z_grid[0]}')
plt.plot(b_grid, n_policy[j-1, :, 1], label=f'z={og.z_grid[1]}')
plt.xlabel('b')
plt.ylabel('n')
plt.title(f'n policy at age {j}')
plt.legend()


# %%
h = SS_distribution(og, b_policy)
# plot histogram of h summing across axis 0 and 2
h_collapsed = h.sum(axis=2).sum(axis=0)
plt.hist(b_grid, bins=40, weights=h_collapsed)
plt.xlabel('b')
plt.ylabel('Density')
plt.title('Steady state distribution of b')


# %%
begin = time.time()
eqbm = solve_SS(og, tol=1e-5, max_iter=1000, rho=.5)
end = time.time()
print(f'Time elapsed: {end-begin} seconds')
print("K, L, r, w: ")
print(eqbm)










# %%
from OLG_TPI_stripped import solve_HH_transition, solve_transition_path

#%%
r_vec = np.repeat(eqbm[2], og.S)
w_vec = np.repeat(eqbm[3], og.S)

b_transition, c_transition, n_transition = solve_HH_transition(og, r_vec, w_vec)

# %%
# plot a policy at age j
j = 10
plt.plot(b_grid, b_transition[j-1, :, 0], label=f'z={og.z_grid[0]}')
plt.plot(b_grid, b_transition[j-1, :, 1], label=f'z={og.z_grid[1]}')
plt.xlabel('b')
plt.ylabel('b\'')
plt.title(f'b policy at age {j}')
plt.legend()

# %%
# test if b_transition is the same as b_policy
b_policy, c_policy, n_policy = solve_HH_SS(og, r=eqbm[2], w=eqbm[3])
np.allclose(b_policy, b_transition)

# %%
og2 = OLGModel(b_grid=b_grid, eta=eta, z_grid=np.array([1.0, .25]), gamma=2.5, theta=.25)
K_0, L_0, r_0, w_0 = solve_SS(og2, tol=1e-5, max_iter=1000, rho=.5)
b_2, c_2, n_2 = solve_HH_SS(og2, r=r_0, w=w_0)
h_0 = SS_distribution(og2, b_2)

# %%
T = og.S * 3
begin = time.time()
transition_path = solve_transition_path(og, T, K_0, L_0, h_0, tol=1e-4, maxiter=500, rho=0.5)
end = time.time()
print(f'Time elapsed: {end-begin} seconds')

# %%

K_path, L_path, r_path, w_path = transition_path
plt.plot(K_path, label='transition path')
# add horizontal line at K_ss
plt.axhline(y=eqbm[0], color='r', linestyle='--', label='Steady State')
plt.xlabel('Time')
plt.ylabel('Capital Stock')
plt.title('Transition path of Capital Stock')
plt.legend()
# %%

plt.plot(L_path, label='transition path')
# add horizontal line at L_ss
plt.axhline(y=eqbm[1], color='r', linestyle='--', label='Steady State')
plt.xlabel('Time')
plt.ylabel('Labor Supply')
plt.title('Transition path of Labor Supply')
plt.legend()


# %%
plt.plot(r_path, label='transition path')
# add horizontal line at r_ss
plt.axhline(y=eqbm[2], color='r', linestyle='--', label='Steady State')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.title('Transition path of Interest Rate')
plt.legend()

# %%
plt.plot(w_path, label='transition path')
# add horizontal line at w_ss
plt.axhline(y=eqbm[3], color='r', linestyle='--', label='Steady State')
plt.xlabel('Time')
plt.ylabel('Wage')
plt.title('Transition path of Wage')
plt.legend()


