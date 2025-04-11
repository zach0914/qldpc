import numpy as np
import stim
from bp_osd import BPOSD
from src import Simulator

from bp_osd import BPOSD
from tqdm import tqdm

def Simulation():
    # noises = range(1, 11, 9)
    # logical_error_rate = []
    # for noise in tqdm(noises):
    ka, kb = 3, 5
    s = Simulator(ka, kb)
    s.create_initial_circuit(0.2)
    c = s.circuit
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

print(Simulation())
# noises, logical_error_rate = simulator()