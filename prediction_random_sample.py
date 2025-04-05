import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import basic

def P(epsilon):
    """Calculate the parity of the number of -1 in the list."""
    count_neg_ones = epsilon.count(-1)
    return -1 if count_neg_ones % 2 != 0 else 1

# calculate partial derivative of order k for F by random sampling
def compute_partial(d, Pauli, k, sample_size, l, i):
    index = np.zeros(k, dtype=int)
    for j in range(k):
        index[j] = i // l**(k-j-1) % l
    result = 0
    for _ in range(sample_size):
        epsilon = np.random.choice([-1, 1], k)
        parity = P(list(epsilon))
        Theta = np.zeros(l)
        for j in range(k):
            Theta[int(index[j])] += epsilon[j]
        F = basic.run_simulation(basic.generate_circuit(d, Theta, Pauli), d)
        result += parity * F
    return i, result / sample_size

def calculate_partial_derivate_random_sample(d, Pauli, k, N_k, sample_size):
    l = len(Pauli)
    partial_F_k = []

    if N_k < l**k:
        indexes = np.random.choice(l**k, N_k, replace=False)
    else:
        indexes = range(l**k)

    with ProcessPoolExecutor() as executor:
        args = [(d, Pauli, k, sample_size, l, i) for i in indexes]
        futures = [executor.submit(compute_partial, *arg) for arg in args]
        for future in as_completed(futures):
            i, value = future.result()
            partial_F_k.append((i, value))

    return partial_F_k

# the function to predict F's k-th order part
def F_random_sample_k(l, k, Theta, N_k, partial_F_k):
    F = 0
    SD = 0
    for i, value in partial_F_k:
        index = np.zeros(k)
        for j in range(k):
            index[j] = i // l**(k-j-1) % l
        F += value * np.prod([Theta[int(index[j])] for j in range(k)])/math.factorial(k) * l**k/N_k
        SD += (value * np.prod([Theta[int(index[j])] for j in range(k)])/math.factorial(k) * l**k)**2/N_k
    SD = math.sqrt(SD - F**2)
    
    return F, SD

# return the list of F_k and SD_k
def F_random_sample(d, Pauli, K, Theta, N_K, sample_size):
    l = len(Pauli)
    F_K = []
    SD_K = []
    for k in range(1,K+1):
        partial_F_K = calculate_partial_derivate_random_sample(d, Pauli, k, N_K[k-1], sample_size)
        f, sd = F_random_sample_k(l, k, Theta, N_K[k-1], partial_F_K)
        F_K.append(f)
        SD_K.append(sd)
        
    return  F_K, SD_K

# function to determine N_K
def determine_N_K(d, Pauli, K, N, theta, method='pre-sample', N_presample = 10**4):
    N_K = []
    
    def pre_sample(d, Pauli, k, N_presample):
        l = len(Pauli)
        result = 0
        index = np.zeros(k)
        for _ in range(N_presample):
            i = np.random.randint(0, l**k)
            for j in range(k):
                index[j] = i // l**(k-j-1) % l
            epsilon = np.random.choice([-1, 1], k)
            parity = P(list(epsilon))
            Theta = np.zeros(l)
            for j in range(k):
                Theta[int(index[j])] += epsilon[j]
            F = basic.run_simulation(basic.generate_circuit(d, Theta, Pauli), d)
            result += parity * F
        return abs(result / N_presample)
    
    def pre_sample_distribution(d, Pauli, K, N, N_presample):
        l = len(Pauli)
        pre_sample_list = []
        for i in range(K):
            pre_sample_list.append((l * theta * np.pi) ** (i+1) * pre_sample(d, Pauli, (i+1), N_presample)/math.factorial(i+1))
        total_sum = np.sum(pre_sample_list)
        return [int(N * i / total_sum) for i in pre_sample_list]

    switcher = {
        'uniform': lambda x: [int(N/K) for _ in range(K)],
        'pre-sample': lambda x: pre_sample_distribution(d, Pauli, K, N, N_presample),
        # add other methods here
    }
    
    N_K = switcher[method](N)
    return N_K

# main function
def F_random_sample_main(d, Pauli, K, Theta, N, sample_size, theta, method='pre-sample', N_presample = 10**4):
    N_K = determine_N_K(d, Pauli, K, N, theta, method, N_presample)
    return F_random_sample(d, Pauli, K, Theta, N_K, sample_size)