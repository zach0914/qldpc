import numpy as np
from qldpc.decoders import Decoder
import numpy as np
import sympy
import stim
import ldpc
from qldpc.codes.quantum import BBCode
from qldpc.decoders import get_decoder_BP_OSD

def generate_cyclic_shift_matrix(size, power):
    """生成循环移位矩阵的幂次"""
    shift_matrix = np.roll(np.eye(size), -1, axis=1)  # 基础循环移位矩阵
    return np.linalg.matrix_power(shift_matrix, power)


def generate_BBCode(l, m):
    """生成 BB code的Parity Check Matrix"""
    x, y = sympy.symbols('x y')
    A1, A2, A3 = x ** 3, y, y ** 2
    B1, B2, B3 = y ** 3, x, x ** 2
    poly_a = A1 + A2 + A3
    poly_b = B1 + B2 + B3
    orders = {x:l, y:m}  
    bb_code = BBCode(orders, poly_a, poly_b, field=2)
    # print("Constructed BBCode:")
    # print(" Physical qubits (n):", bb_code.num_qubits)
    # print(" Logical qubits (k):", bb_code.dimension)
    # print(f" parity check matrix:\n{bb_code.matrix}")
    # print(" X parity check matrix (matrix_x):")
    # print(bb_code.matrix_x)
    # print(" Z parity check matrix (matrix_z):")
    # print(bb_code.matrix_z)
    X = np.kron(generate_cyclic_shift_matrix(l, 1), np.eye(m))
    Y = np.kron(np.eye(l), generate_cyclic_shift_matrix(m, 1))
    # A1_np = X ** 3
    # A2_np = Y
    # A3_np = Y ** 2
    # B1_np = Y ** 3
    # B2_np = X
    # B3_np = X ** 2
    return bb_code

from typing import List
def get_destabilizers(
        stabilizer_generators: List[stim.PauliString],
) -> List[stim.PauliString]:
    t = stim.Tableau.from_stabilizers(stabilizer_generators, allow_redundant=True, allow_underconstrained=True)
    return [t.x_output(k) for k in range(len(t))]

def generate_error(Theta, Pauli):
    """生成错误"""
    result = stim.Circuit()
    assert len(Theta) == len(Pauli)
    
    operations = {
        'X': {
            1: ['S', 'Z', 'H', 'Z', 'S'],
            2: ['X'],
            3: ['S', 'H', 'S']
        },
        'Y': {
            1: ['H', 'Z'],
            2: ['Y'],
            3: ['Z', 'H']
        },
        'Z': {
            1: ['H', 'S', 'Z', 'H', 'Z', 'S', 'H'],
            2: ['Z'],
            3: ['H', 'S', 'H', 'S', 'H']
        }
    }
    
    for i in range(len(Theta)):
        Theta[i] = Theta[i] % 4
        if Theta[i] == 0:
            continue
        
        pauli_type = Pauli[i][0]
        qubit = Pauli[i][1]
        for op in operations[pauli_type].get(Theta[i], []):
            result.append(op, qubit)
        
    return result


def generate_syndrome_circuit(bb_code, default = True):
    """syndrome measurement circuit, including detectors"""
    # 提取BB code信息
    num_data_qubits = bb_code.num_qubits
    num_ancilla_qubits = num_data_qubits
    num_logical_qubits = bb_code.dimension
    HX = bb_code.matrix_x
    HZ = bb_code.matrix_z 
    # print(f"HX.shape = {HX.shape}")
    Pauli = [('Z', i) for i in range(num_data_qubits)]
    Theta = np.ones(num_data_qubits) - np.kron(np.ones(num_data_qubits // 2), np.array([0, 1]))#np.random.rand(num_data_qubits)
    

    ## 利用BB code提供的parity check matrix生成不含任何测量的circuit
    pauli_strings = []
    for row in range(HX.shape[0]):
        pauli_string = stim.PauliString("I" * num_data_qubits)  # 初始化为全 I
        for col in range(num_data_qubits):
            if HX[row][col] == 1:
                pauli_string[col] = 'X'
        pauli_strings.append(pauli_string)
    for row in range(HZ.shape[0]):
        pauli_string = stim.PauliString("I" * num_data_qubits)  # 初始化为全 I
        for col in range(num_data_qubits):
            if HZ[row][col] == 1:
                pauli_string[col] = 'Z'
        pauli_strings.append(pauli_string)
    destabilizers = get_destabilizers(pauli_strings)
    logical_operator = np.zeros((num_data_qubits, num_data_qubits))
    for i in range(num_data_qubits):
        for j, p in enumerate(str(destabilizers[i])):
            if p == 'Z':
                logical_operator[i, j - 1] = 1
            elif p == 'X':
                logical_operator[i, j - 1] = -1
    tableau = stim.Tableau.from_stabilizers(pauli_strings, allow_redundant=True, allow_underconstrained=True)
    circuit = tableau.to_circuit(method="elimination")

    # 添加错误（不确定是不是应该这么加）
    if default == True:
        error_prob = np.sin(0.01 * np.pi)**2
        for qubit in range(num_data_qubits):
            circuit.append('Z_ERROR', [qubit], 0.5)
    else:
        circuit += generate_error(Theta, Pauli) 
    # 添加稳定子测量和检测器
    ### 添加X稳定子
    ancillas = list(range(num_data_qubits, num_data_qubits + num_ancilla_qubits))  # 辅助量子位7-12
    for i in range(HX.shape[0]):
        gen = HX[i]
        a = ancillas[i]
        circuit.append("H", [a])
        for q in range(num_data_qubits):
            if gen[q] == 1:
                circuit.append("CX", [a, q])
        circuit.append("H", [a])
    
        circuit.append("M", [a])
        circuit.append("DETECTOR", [stim.target_rec(-1)])  # 记录当前测量结果作为检测器
        circuit.append("R", [a])  # 重置辅助位
        circuit.append("TICK")

    
    ### 添加Z稳定子
    for i in range(HZ.shape[0]):
        gen = HZ[i]
        a = ancillas[i + HX.shape[0]]
        circuit.append("H", [a])
        for q in range(num_data_qubits):
            if gen[q] == 1:
                circuit.append("CZ", [a, q])
        circuit.append("H", [a])
        
        circuit.append("M", [a])
        circuit.append("DETECTOR", [stim.target_rec(-1)])  # 记录当前测量结果作为检测器
        circuit.append("R", [a])  # 重置辅助位
        circuit.append("TICK")
    # 最后测量数据量子位
    measurements = []  # 用来保存所有测量结果
    for i in range(num_data_qubits):
        if logical_operator[0, i] == 1:
            # 向电路添加测量操作
            circuit.append("M", [j])
            measurements.append(stim.target_rec(-1))  # 将测量结果记录下来
    circuit.append("OBSERVABLE_INCLUDE", measurements, 0)

    # for i in range(num_data_qubits):
    #     measurements = []  # 用来保存所有测量结果
    #     for j in range(num_data_qubits):
    #         if logical_operator[i, j] == 1:
    #             # 向电路添加测量操作
    #             circuit.append("M", [j])
    #         elif logical_operator[i, j] == -1:
    #             circuit.append("H", [j])
    #             circuit.append("M", [j])
    #             circuit.append("H", [j])
    #         measurements.append(stim.target_rec(-1))  # 将测量结果记录下来
    #     # 将所有测量结果加入 OBSERVABLE_INCLUDE
    #     # 确保所有测量结果都包括在内
    #     circuit.append("OBSERVABLE_INCLUDE", measurements, i)
    return circuit


import pymatching 
from bp_osd import BPOSD
def main():
    # ========= 1. 构造 BBCode =========
    # 利用多项式参数构造 BB code（参数仅为示例，可根据需求修改）
    l, m = 6, 6
    bb_code = generate_BBCode(l, m)
    
    # Generate syndrome measurement circuit
    default_circuit = generate_syndrome_circuit(bb_code)
    # Print the circuit
    # print("Syndrome measurement circuit:")
    # circuit.diagram('timeline-svg')

    # # find the dem of the original circuit
    dem = default_circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    decoder = BPOSD(dem, max_bp_iters = 20) #使用BP_OSD
    print(repr(dem))
    dem.diagram("matchgraph-svg")

    circuit = generate_syndrome_circuit(bb_code)
    
    # sample 100 shots
    N = 100
    sampler = circuit.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=N, separate_observables=True)
            
    # 打印 syndrome 的一些信息
    print("Syndrome shape:", syndrome.shape)
    print("First few syndromes:", syndrome[:5])

    predicted_observables = decoder.decode_batch(syndrome)
    print(f"actual_observable:{actual_observables}\npredicted_observables:{predicted_observables}")

    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

    return num_errors / N
print(main())


