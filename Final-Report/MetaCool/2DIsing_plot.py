import numpy as np
import matplotlib.pyplot as plt


def analytic_M(T_p):
    T_p_c = 2.269185
    z = np.exp(- 2 / T_p)
    if T_p <= T_p_c:
        return np.power(1 + z**2, 1/4) * np.power(1 - 6 * z**2 + z**4, 1/8) / np.sqrt(1 - z**2)
    else:
        return 0


# number of spins on each row and column of the 2D array
Ns = np.array([50])

# temperature times the Boltzmann constant divided by J, T_p = k * T / J
T_ps = np.arange(0, 5.05, 0.05)

# number of steps taken in the Markov chain
steps = [int(1e5), int(1e7), int(2e6)]
Hsteps = int(1e6)

# an additional term to the energy divided by J due to an external magnetic field, H_p = H / J
H_ps = np.array([-0.5, -0.1, 0., 0.1, 0.5])


M_as = []
for T_p in T_ps[1:]:
    M_as.append(analytic_M(T_p))
M_as = np.array(M_as)

for N in Ns:
    fig, ax = plt.subplots()
    for H_p in H_ps:
        if H_p == 0:
            E_ps = np.load(f'Data/E_ps_N{N}_H_p{H_p}_steps{steps[0]}-{steps[1]}-{steps[2]}.npy')
        else:
            E_ps = np.load(f'Data/E_ps_N{N}_H_p{H_p}_steps{Hsteps}.npy')
            
        ax.plot(T_ps, E_ps, marker='+', linestyle='None', label=r'$H^{\prime}' + f' = {H_p}$')
        
    ax.legend()
    ax.set_xlabel(r'$T^{\prime}$')
    ax.set_ylabel(r'$E^{\prime}$')
    ax.set_title(f'{N}' + r'$\times$' + f'{N} spins')
    fig.tight_layout()
    plt.savefig(f'Figs/energy_N{N}_steps{steps[0]}-{steps[1]}-{steps[2]}_Hsteps{Hsteps}.png', bbox_inches='tight')
    plt.show()
    
for N in Ns:
    fig, ax = plt.subplots()
    for H_p in H_ps:
        if H_p == 0:
            M_norms = np.load(f'Data/M_norms_N{N}_H_p{H_p}_steps{steps[0]}-{steps[1]}-{steps[2]}.npy')
        else:
            M_norms = np.load(f'Data/M_norms_N{N}_H_p{H_p}_steps{Hsteps}.npy')
            
        ax.plot(T_ps, M_norms, marker='+', linestyle='None', label=r'$H^{\prime}' + f' = {H_p}$')
        
    ax.plot(T_ps[1:], M_as, label='analytic')
    ax.legend()
    ax.set_xlabel(r'$T^{\prime}$')
    ax.set_ylabel(r'$M$')
    ax.set_title(f'{N}' + r'$\times$' + f'{N} spins')
    fig.tight_layout()
    plt.savefig(f'Figs/magnetization_N{N}_steps{steps[0]}-{steps[1]}-{steps[2]}_Hsteps{Hsteps}.png', bbox_inches='tight')
    plt.show()

