import numpy as np
import matplotlib.pyplot as plt
from Ising import Solve2DIsing, randspin


# number of spins on each row and column of the 2D array
Ns = np.array([50])

# temperature times the Boltzmann constant divided by J, T_p = k * T / J
T_ps = np.arange(0, 5.05, 0.05)[::-1]

# number of steps taken in the Markov chain
steps = [int(2e6), int(1e7)]
Hsteps = int(5e6)

# an additional term to the energy divided by J due to an external magnetic field, H_p = H / J
H_ps = np.array([0.5, 0.1, -0.1, -0.5])


for N in Ns:
    for H_p in H_ps:
        S = randspin(N)
        E_ps = []
        Ms = []
        for T_p in T_ps:
            if H_p == 0:
                if 2.1 <= T_p <= 2.5:
                    E_p, M, S_new = Solve2DIsing(S, T_p, steps[1], H_p)
                else:
                    E_p, M, S_new = Solve2DIsing(S, T_p, steps[0], H_p)
            else:
                E_p, M, S_new = Solve2DIsing(S, T_p, Hsteps, H_p)
                
            E_ps.append(E_p)
            Ms.append(M)
            S = S_new

        E_ps = np.array(E_ps)
        Ms = np.array(Ms)
        M_norms = Ms / S.size

        if H_p == 0:
            np.save(f'Data/cool_E_ps_N{N}_H_p{H_p}_steps{steps[0]}-{steps[1]}.npy', E_ps)
            np.save(f'Data/cool_Ms_N{N}_H_p{H_p}_steps{steps[0]}-{steps[1]}.npy', Ms)
            np.save(f'Data/cool_M_norms_N{N}_H_p{H_p}_steps{steps[0]}-{steps[1]}.npy', M_norms)
        else:
            np.save(f'Data/cool_E_ps_N{N}_H_p{H_p}_steps{Hsteps}.npy', E_ps)
            np.save(f'Data/cool_Ms_N{N}_H_p{H_p}_steps{Hsteps}.npy', Ms)
            np.save(f'Data/cool_M_norms_N{N}_H_p{H_p}_steps{Hsteps}.npy', M_norms)


fig, ax = plt.subplots()
ax.scatter(T_ps, E_ps, marker='+')
ax.set_xlabel(r'$T^{\prime} = \frac{kT}{J}$')
ax.set_ylabel(r'$E^{\prime} = \frac{E}{J}$')
fig.tight_layout()
# plt.savefig(f'Figs/cool_E_ps_N{N}_H_p{H_p}.png', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
ax.scatter(T_ps, M_norms, marker='+')
ax.set_xlabel(r'$T^{\prime} = \frac{kT}{J}$')
ax.set_ylabel(r'$M$')
fig.tight_layout()
# plt.savefig(f'Figs/cool_M_norms_N{N}_H_p{H_p}.png', bbox_inches='tight')
plt.show()

