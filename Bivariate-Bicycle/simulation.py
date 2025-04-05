import numpy as np
import stim
from bp_osd import BPOSD
from src import Simulator

from bp_osd import BPOSD
from tqdm import tqdm

# k = 8
# code = [12,3,9,1,2,0,1,11,3,12,2,0,11,6]
code = [9,5,8,4,1,5,8,7,5,9,2,1,9,5]
# code = [12,5,10,4,1,0,1,2,5,12,2,0,11,6]
#code = [15,5,5,2,3,2,7,6,15,5,0,4,11,6]
#code = [30,3,25,3,1,3,22,13,3,30,3,4,11,9]

# k = 12
#code = [14,7,6,5,6,0,4,13,7,14,0,2,12,8]
#code = [18,6,3,4,5,3,7,2,6,18,1,4,16,15]
# code = [12,6,3,1,2,3,1,2,12,6,1,4,16,15]

# code = [14,7,3,5,2,7,6,9,7,14,3,3,11,5]

# k = 16
#code = [15,5,10,3,2,0,9,7,5,15,3,4,16,15]
#code = [30,5,5,3,2,5,3,29,5,30,3,4,17,15]

def Simulation(code, Theta, Pauli):
    lr_time = 1
    
    T = 1
    # noises = range(1, 11, 9)
    # logical_error_rate = []
    # for noise in tqdm(noises):
    s = Simulator(code)
    s.create_initial_circuit(T, lr_time, 0.01, only_coherent_error=True)
    s.simulate(Theta, Pauli, only_coherent_error=True)
    c = s.c
    # print(c)

    sampler = c.compile_detector_sampler()
    dem = c.detector_error_model()
    
    decoder = BPOSD(dem, max_bp_iters = 20) #使用BP_OSD

    # # sample 10 shots
    N = 50
    syndrome, actual_observables = sampler.sample(shots=N, separate_observables=True)

    # print("Syndrome shape:", syndrome.shape)
    # print("First few syndromes:", syndrome[:5])

    predicted_observables = decoder.decode_batch(syndrome)
    # print(f"actual_observable:{actual_observables}\npredicted_observables:{predicted_observables}")

    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    # logical_error_rate.append(num_errors / N)
    print(f'Error rate: {num_errors / N * 100}%')
    return num_errors / N# logical_error_rate

theta_list = np.linspace(0, 0.3, 100)
theta = theta_list[0]
Pauli = [('Z', i) for i in range(code[0] * code[1] * 4)]
Theta = np.array([theta for _ in range(len(Pauli))])
print(Simulation(code, Theta, Pauli))
# noises, logical_error_rate = simulator()