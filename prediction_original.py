import numpy as np
import itertools
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import basic

def P(epsilon):
    """Calculate the parity of the number of -1 in the list."""
    count_neg_ones = epsilon.count(-1)
    return -1 if count_neg_ones % 2 != 0 else 1

# calculate partial derivative of order k for F
def calculate_partial_derivative(d, Pauli, k):
    l = len(Pauli)
    partial_F_K = np.zeros(l**k)

    def compute_partial(i):
        index = np.zeros(k, dtype=int)
        for j in range(k):
            index[j] = i // l**(k-j-1) % l
        # 仅计算index中元素从小到大的情况
        index_ = np.sort(index)
        if not np.array_equal(index, index_):
            return i, 0
        result = 0
        for epsilon in itertools.product([-1, 1], repeat=k):
            parity = P(list(epsilon))
            Theta = np.zeros(l)
            for j in range(k):
                Theta[int(index[j])] += epsilon[j]
            result += parity * basic.run_simulation(basic.generate_circuit(d, Theta, Pauli), d)
        return i, result / 2**k

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_partial, i) for i in range(l**k)]
        for future in as_completed(futures):
            i, value = future.result()
            partial_F_K[i] = value

    return partial_F_K

# the function to predict F
def F(d, Pauli, K, Theta):
    l = len(Pauli)
    result = 0
    for k in range(1,K+1):
        partial_F_K = calculate_partial_derivative(d, Pauli, k)
        for i in range(l**k):
            index = np.zeros(k)
            for j in range(k):
                index[j] = i // l**(k-j-1) % l
            index_ = np.sort(index)
            i_ = 0
            for j in range(k):
                i_ += index_[j] * l**(k-j-1)
            result += partial_F_K[i_] * np.prod([Theta[int(index[j])] for j in range(k)])/math.factorial(k)
    return result