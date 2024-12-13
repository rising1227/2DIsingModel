import numpy as np
import matplotlib.pyplot as plt


def energy(S, H_p=0):
    """Compute the total energy (divided by J) for a given configuration of the 2D Ising model."""
    
    S_down = np.roll(S, -1, axis=0)
    S_right = np.roll(S, -1, axis=1)
    
    return -(np.sum(S * S_down) + np.sum(S * S_right)) - H_p * np.sum(S)


def flip(s):
    """Flip a spin."""
    return -s


def energy_chg(S, i, j, H_p=0):
    """Compute the change of energy (divided by J) due to a spin flip at (i, j)."""
    
    N = S.shape[0]
    newspin = flip(S[i, j])
    
    chg = 0

    chg += (- newspin * S[i-1, j]) - (- S[i, j] * S[i-1, j])
    chg += (- newspin * S[(i+1) % N, j]) - (- S[i, j] * S[(i+1) % N, j])
    chg += (- newspin * S[i, j-1]) - (- S[i, j] * S[i, j-1])
    chg += (- newspin * S[i, (j+1) % N]) - (- S[i, j] * S[i, (j+1) % N])
    
    chg += (- H_p * newspin) - (- H_p * S[i, j])
    
    return chg


def Solve2DIsing(S, T_p, steps, H_p=0, test=False):
    """
    Compute the energy (divided by J), magnetization, and configuration of the 2D Ising model 
    at a given temperature (times the Boltzmann constant divided by J).
    """
    E_p = energy(S, H_p)
    M = np.sum(S)
    E_p_set = []
    M_set = []
    E_p_set.append(E_p)
    M_set.append(M)
    
    N = S.shape[0]
    
    rng = np.random.default_rng(12345)
    
    for k in range(steps):
        i = rng.integers(N)
        j = rng.integers(N)
        
        dE_p = energy_chg(S, i, j, H_p)
        
        if T_p == 0:
            if dE_p <= 0:
                E_p += dE_p
                M += (flip(S[i, j]) - S[i, j])
                S[i, j] = flip(S[i, j])
        else:
            if rng.random() < np.exp(- dE_p / T_p):
                E_p += dE_p
                M += (flip(S[i, j]) - S[i, j])
                S[i, j] = flip(S[i, j])
                
        E_p_set.append(E_p)
        M_set.append(M)
    
    E_p_set = np.array(E_p_set)
    M_set = np.array(M_set)
    
    # for testing
    if test:
        fig, ax = plt.subplots()
        ax.plot(E_p_set)
        ax.set_xlabel('steps')
        ax.set_ylabel(r'$E^{\prime}$')
        ax.set_title(f'N{N}_T_p{T_p}_H_p{H_p}')
        plt.show()
        
        fig, ax = plt.subplots()
        ax.plot(M_set / S.size)
        ax.set_xlabel('steps')
        ax.set_ylabel(r'$M$')
        ax.set_title(f'N{N}_T_p{T_p}_H_p{H_p}')
        plt.show()

    E_p_avg = np.average(E_p_set[int(steps/2):])
    M_avg = np.average(M_set[int(steps/2):])
    
    return E_p_avg, M_avg, S


def randspin(N):
    """Generate a random configuration of the 2D Ising model."""
    rng = np.random.default_rng(12345)
    A = rng.integers(2, size=(N, N))
    return 2 * A - 1

