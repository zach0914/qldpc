import stim
import numpy as np
from numpy.linalg import matrix_power, matrix_rank
import matplotlib.pyplot as plt
from mec import make_circle
from scipy.sparse import lil_matrix
import os
from tqdm import tqdm
import galois
from utils import cyclic_shift_matrix, par2gen, commute, SGSOP, get_nbr, get_logicals, manhattan, generate_error
#ml = n/2 for X check, ml = n/2 for Z check

# 0.001 error rate, everything other than CNOT and measurement is 10x less
# muls001 = np.array([0,0,1,0,2.83437563,0,2.71543645,0,2.66390109,0,2.47514626,0,2.63554603,0,2.8225308,0,3.20738319,0,3.05384743,0,3.01289603,0,3.30535152,0,3.46853617,0,3.30472058,0,3.34178504,0,3.69494846,0,3.74955935,0,3.89150943,0,4.29804057,0,4.622713,0,4.64298887,0,4.83355337,0,4.63747826,0,5.44062479,0,5.48444052,0,6.05625242,0,5.75421291,0,6.06743327,0,6.63790984,0,6.33608196,0,7.24417597,0,6.69331914,0,7.44078094,0,7.59173345,0,8.37524776,0,8.38899217,0,8.80930114,0,9.80709627,0,8.87491589,0,9.62782511,0,9.85953381])
# probs = np.array([0,0,1,0,0.9914,0,0.98695,0,0.98352,0,0.97772,0,0.97513,0,0.97076,0,0.96652,0,0.96272,0,0.95921,0,0.953,0,0.95141,0,0.94713,0,0.94261,0,0.93912,0,0.93611,0,0.9328,0,0.92833,0,0.9237,0,0.92182,0,0.91651,0,0.91429,0,0.91166,0,0.9062,0,0.90485,0,0.90021,0,0.89659,0,0.89486,0,0.89014,0,0.88899,0,0.88297,0,0.87894,0,0.87727,0,0.87281,0,0.87138,0,0.86613,0,0.86468,0,0.86198,0,0.85793,0,0.85501])
# idle_error = 0.0002

class Simulator:
    def __init__(self, code):
        self.code = code
        self.ell = code[0]
        self.m = code[1]
        self.muls001 = np.array([0,0,1,0,2.83437563,0,2.71543645,0,2.66390109,0,2.47514626,0,2.63554603,0,2.8225308,0,3.20738319,0,3.05384743,0,3.01289603,0,3.30535152,0,3.46853617,0,3.30472058,0,3.34178504,0,3.69494846,0,3.74955935,0,3.89150943,0,4.29804057,0,4.622713,0,4.64298887,0,4.83355337,0,4.63747826,0,5.44062479,0,5.48444052,0,6.05625242,0,5.75421291,0,6.06743327,0,6.63790984,0,6.33608196,0,7.24417597,0,6.69331914,0,7.44078094,0,7.59173345,0,8.37524776,0,8.38899217,0,8.80930114,0,9.80709627,0,8.87491589,0,9.62782511,0,9.85953381])
        self.probs = np.array([0,0,1,0,0.9914,0,0.98695,0,0.98352,0,0.97772,0,0.97513,0,0.97076,0,0.96652,0,0.96272,0,0.95921,0,0.953,0,0.95141,0,0.94713,0,0.94261,0,0.93912,0,0.93611,0,0.9328,0,0.92833,0,0.9237,0,0.92182,0,0.91651,0,0.91429,0,0.91166,0,0.9062,0,0.90485,0,0.90021,0,0.89659,0,0.89486,0,0.89014,0,0.88899,0,0.88297,0,0.87894,0,0.87727,0,0.87281,0,0.87138,0,0.86613,0,0.86468,0,0.86198,0,0.85793,0,0.85501])
        self.idle_error = 0.0002
        
        # theta_list = np.linspace(0, 0.3, 100)
        # theta = theta_list[0]
        # Pauli = [('Z', i) for i in range(m * self.ell * 4)]
        # Theta = np.array([theta for _ in range(len(Pauli))])

        x = np.kron(cyclic_shift_matrix(self.ell), np.eye(self.m))
        y = np.kron(np.eye(self.ell), cyclic_shift_matrix(self.m))

        self.A1 = matrix_power(x, code[2])
        self.A2 = matrix_power(y, code[3])
        self.A3 = matrix_power(y, code[4])
        self.A = ( self.A1 + self.A2 + self.A3 ) % 2

        self.B1 = matrix_power(y, code[5])
        self.B2 = matrix_power(x, code[6])
        self.B3 = matrix_power(x, code[7])
        self.B = ( self.B1 + self.B2 + self.B3 ) % 2

        self.Hx = np.hstack([self.A, self.B]).astype(int)
        self.Hz = np.hstack([self.B.T, self.A.T]).astype(int)

        GF = galois.GF(2)
        arr = GF(self.Hz.T)
        k = 2 * (self.Hz.T.shape[1] - matrix_rank(arr))
        def embed_code(code, init):
            emb_m, emb_ell, A_ind, B_ind = code

            lattice = np.empty((2*emb_m, 2*emb_ell), dtype=object)
            lattice[0][0] = f"x{init}"

            # As = [[A1, A2.T], [A2, A3.T], [A1, A3.T]]
            # Bs = [[B1, B2.T], [B2, B3.T], [B1, B3.T]]
            As = [[self.A1, self.A2.T], [self.A2, self.A1.T], [self.A2, self.A3.T], [self.A3, self.A2.T], [self.A1, self.A3.T], [self.A3, self.A1.T]]
            Bs = [[self.B1, self.B2.T], [self.B2, self.B1.T], [self.B2, self.B3.T], [self.B3, self.B2.T], [self.B1, self.B3.T], [self.B3, self.B1.T]]

            for i in range(2*emb_m - 1):
                for j in range(2*emb_ell):
                    curr_ind = int(lattice[i][j][1:])

                    if (i % 2 == 0):
                        tmp_A = As[A_ind][1]
                    else:
                        tmp_A = As[A_ind][0]
                    if (j % 2 == 0):
                        tmp_B = Bs[B_ind][1]
                    else:
                        tmp_B = Bs[B_ind][0]

                    lattice[(i+1)%(2*emb_m)][j] = f"{get_nbr((i+1)%(2*emb_m), j)}{np.where(tmp_A @ np.eye(self.m*self.ell)[curr_ind])[0][0]}"
                    lattice[i][(j+1)%(2*emb_ell)] = f"{get_nbr(i, (j+1)%(2*emb_ell))}{np.where(tmp_B @ np.eye(self.m*self.ell)[curr_ind])[0][0]}"

            for i in range(2*emb_m):
                for j in range(2*emb_ell):
                    if (lattice[i][j][0] == "z"):
                        lattice[i][j] = f"z{int(lattice[i][j][1:]) + self.m*self.ell}"
                    elif (lattice[i][j][0] == "r"):
                        lattice[i][j] = f"r{int(lattice[i][j][1:]) + self.m*self.ell}"

            return lattice
        
        lattice = embed_code((code[8],code[9],code[10],code[11]), 0)
        self.all_qbts = {} # 存储全部2n个qubits

        self.qbts = np.array([None for i in range(2*self.m*self.ell)]) # 存储n个data qubit的坐标

        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i][j][0] == "r" or lattice[i][j][0] == "l":
                    self.all_qbts[(i,j)] = int(lattice[i][j][1:])
                    self.qbts[int(lattice[i][j][1:])] = (i, j)
        self.x_checks = np.array([None for i in range(self.m*self.ell)])
        self.z_checks = np.array([None for i in range(self.m*self.ell)])        
        
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                if lattice[i][j][0] == "x":
                    self.all_qbts[(i,j)] = int(lattice[i][j][1:]) + 2*self.m*self.ell
                    self.x_checks[int(lattice[i][j][1:])] = (i, j)
                elif lattice[i][j][0] == "z":
                    self.all_qbts[(i,j)] = int(lattice[i][j][1:]) + 2*self.m*self.ell
                    self.z_checks[int(lattice[i][j][1:])-(self.m*self.ell)] = (i, j)
        self.x_rs = []
        self.z_rs = []
        for i in range(self.m*self.ell):
            gen_qbts = self.qbts[np.where(self.Hx[i])[0]]
            self.x_rs.append(make_circle(gen_qbts)[2]) # return the radius of the cycle
        for i in range(self.m*self.ell):
            gen_qbts = self.qbts[np.where(self.Hz[i])[0]]  # return the radius of the cycle
            self.z_rs.append(make_circle(gen_qbts)[2])
        self.lr_x_checks = np.array([], dtype=int)
        self.sr_x_checks = np.array([], dtype=int)
        self.lr_z_checks = np.array([], dtype=int)
        self.sr_z_checks = np.array([], dtype=int)

        # 模拟syndrome measurement成功概率
        self.z_check_succ_probs = np.ones(self.m*self.ell)
        self.x_check_succ_probs = np.ones(self.m*self.ell)

        
        for i, x_check in enumerate(self.x_checks):
            gen_qbts = self.qbts[np.where(self.Hx[i])[0]]

            nonlocal_qbts = []
            if (self.x_rs[i] > (min(self.x_rs)+np.std(self.x_rs))):
                self.lr_x_checks = np.append(self.lr_x_checks, i)
            else:
                self.sr_x_checks = np.append(self.sr_x_checks, i)

            # for qbt in gen_qbts:
                # x_check_succ_probs[i] *= probs[manhattan([x_check, qbt])+1]

        # print(x_check_succ_probs)

        for i, z_check in enumerate(self.z_checks):
            gen_qbts = self.qbts[np.where(self.Hz[i])[0]]

            nonlocal_qbts = []
            if (self.z_rs[i] > min(self.z_rs)+np.std(self.z_rs)):
                self.lr_z_checks = np.append(self.lr_z_checks, i)
            else:
                self.sr_z_checks = np.append(self.sr_z_checks, i)

            # for qbt in gen_qbts:
                # z_check_succ_probs[i] *= probs[manhattan([z_check, qbt])+1] 

        adv = sum(np.array(self.x_rs)[self.lr_x_checks]) / sum(self.x_rs)

    def measure_x_checks(self, checks, p, scale=False, only_coherent_error=False):
        c = stim.Circuit()
        c.append("H", [self.all_qbts[self.x_checks[x_check]] for x_check in checks])
        if only_coherent_error == False:
            c.append("DEPOLARIZE1", [self.all_qbts[self.x_checks[x_check]] for x_check in checks], p/10)
        for x in checks:
            gen_qbts = self.qbts[np.where(self.Hx[x])[0]]
            for qbt in gen_qbts:
                path_qbts = [self.all_qbts[self.x_checks[x]], self.all_qbts[qbt]]
                c.append("CNOT", path_qbts)
                if only_coherent_error == False:
                    if scale:
                        c.append("DEPOLARIZE2", path_qbts, p*self.muls001[manhattan([self.x_checks[x], qbt])+1])
                    else:
                        c.append("DEPOLARIZE2", path_qbts, p)
        c.append("H", [self.all_qbts[self.x_checks[x_check]] for x_check in checks])
        # c.append("DEPOLARIZE1", [all_qbts[x_checks[x_check]] for x_check in checks], p/10)
        return c

    def measure_z_checks(self, checks, p, scale=False, only_coherent_error =False):
        c = stim.Circuit()
        for z in checks:
            gen_qbts = self.qbts[np.where(self.Hz[z])[0]]
            for qbt in gen_qbts:
                path_qbts = [self.all_qbts[qbt], self.all_qbts[self.z_checks[z]]]
                c.append("CNOT", path_qbts)
                if only_coherent_error == False:
                    if scale:
                        c.append("DEPOLARIZE2", path_qbts, p*self.muls001[manhattan([qbt, self.z_checks[z]])+1])
                    else:
                        c.append("DEPOLARIZE2", path_qbts, p)
        return c

    def all_checks(self, maximum_error_rate = 0, scale = False, only_coherent_error = False):
        c = stim.Circuit()
        c += self.measure_z_checks(self.sr_z_checks, maximum_error_rate, scale, only_coherent_error)
        c += self.measure_z_checks(self.lr_z_checks, maximum_error_rate, scale, only_coherent_error)
        c += self.measure_x_checks(self.sr_x_checks, maximum_error_rate, scale, only_coherent_error)
        c += self.measure_x_checks(self.lr_x_checks, maximum_error_rate, scale, only_coherent_error)
        return c
        
    def create_initial_circuit(self, num_rounds, lr_time, maximum_error_rate = 0.01, only_coherent_error = False):
        self.num_rounds = num_rounds
        self.lr_time = lr_time

        self.prev_meas_z = np.arange(1, self.m*self.ell+1, dtype=int)
        self.prev_meas_x = np.arange(self.m*self.ell+1, 2*self.m*self.ell+1,  dtype=int)
        self.curr_meas_z = np.zeros(self.m*self.ell, dtype=int)
        self.curr_meas_x = np.zeros(self.m*self.ell, dtype=int)

        self.route_confirmation_z = np.ones(self.m*self.ell, dtype=int)
        # self.route_confirmation_z[lr_z_checks] = 0
        self.route_confirmation_x = np.ones(self.m*self.ell, dtype=int)
        # self.route_confirmation_x[lr_x_checks] = 0
        self.detector_history = np.zeros(self.m*self.ell)

        self.p = maximum_error_rate

        self.c = stim.Circuit()
        for key, value in self.all_qbts.items():
            self.c.append("QUBIT_COORDS", value, (key[0],key[1],0))
            self.c.append("QUBIT_COORDS", value+(4*self.m*self.ell), (key[0],key[1],1))
        self.c.append("R", [qbt for qbt in self.all_qbts.values()])
        self.c.append("R", [qbt+(4*self.m*self.ell) for qbt in self.all_qbts.values()])

        self.c += self.all_checks(maximum_error_rate).without_noise()

        self.c.append("MR", [self.all_qbts[z_check] for z_check in self.z_checks])
        self.c.append("MR", [self.all_qbts[x_check] for x_check in self.x_checks])
        if only_coherent_error == False:
            self.c.append("Z_ERROR", [self.all_qbts[qbt] for qbt in self.qbts], 0.001+2*(self.code[12]+self.code[13])*self.idle_error)
    def detectors(self):
        num_meas = self.c.num_measurements
        for i, z_check in enumerate(self.curr_meas_z):
            coord = self.z_checks[i]
            if z_check:
                self.c.append("DETECTOR", [stim.target_rec(self.curr_meas_z[i]-num_meas-1), stim.target_rec(self.prev_meas_z[i]-num_meas-1)], (coord[0], coord[1], 0))
                self.prev_meas_z[i] = self.curr_meas_z[i]
                self.curr_meas_z[i] = 0
        for i, x_check in enumerate(self.curr_meas_x):
            coord = self.x_checks[i]
            if x_check:
                self.c.append("DETECTOR", [stim.target_rec(self.curr_meas_x[i]-num_meas-1), stim.target_rec(self.prev_meas_x[i]-num_meas-1)], (coord[0], coord[1], 0))
                self.prev_meas_x[i] = self.curr_meas_x[i]
                self.curr_meas_x[i] = 0

    def observables(self, type):
        for i, logical in enumerate(get_logicals(self.Hx, self.Hz, type)):
            incl_qbts = np.where(logical)[0]
            incl_qbts = [-j-1 for j in incl_qbts]
            self.c.append("OBSERVABLE_INCLUDE", [stim.target_rec(j) for j in incl_qbts], i)

    # def sr_round(self, with_gate_noise=True, with_synd_noise=True):
    #     curr_sr_z_checks = sr_z_checks[self.route_confirmation_z[sr_z_checks]==1]
    #     curr_sr_x_checks = sr_x_checks[self.route_confirmation_x[sr_x_checks]==1]
    #     self.c += measure_z_checks(curr_sr_z_checks, self.p if with_gate_noise else 0, scale=True)
    #     self.c += measure_x_checks(curr_sr_x_checks, self.p if with_gate_noise else 0, scale=True)

    #     if with_synd_noise: self.c.append("X_ERROR", [all_qbts[z_checks[z_check]] for z_check in curr_sr_z_checks], self.p)
    #     if with_synd_noise: self.c.append("X_ERROR", [all_qbts[x_checks[x_check]] for x_check in curr_sr_x_checks], self.p)

    #     for i, z_check in enumerate(curr_sr_z_checks):
    #         self.c.append("M", all_qbts[z_checks[z_check]])
    #         self.curr_meas_z[z_check] = self.c.num_measurements
    #     for i, z_check in enumerate(sr_z_checks):
    #         self.c.append("R", all_qbts[z_checks[z_check]])
    #     for i, x_check in enumerate(curr_sr_x_checks):
    #         self.c.append("M", all_qbts[x_checks[x_check]])
    #         self.curr_meas_x[x_check] = self.c.num_measurements
    #     for i, x_check in enumerate(sr_x_checks):
    #         self.c.append("R", all_qbts[x_checks[x_check]])

    #     # if with_synd_noise: self.c.append("X_ERROR", [all_qbts[z_checks[z_check]] for z_check in sr_z_checks], self.p/10)
    #     if with_synd_noise: self.c.append("X_ERROR", [all_qbts[x_checks[x_check]] for x_check in sr_x_checks], self.p/10)

    def lr_round(self, with_gate_noise=False, with_synd_noise=False, only_coherent_error = False):
        curr_sr_z_checks = self.sr_z_checks[self.route_confirmation_z[self.sr_z_checks]==1]
        curr_sr_x_checks = self.sr_x_checks[self.route_confirmation_x[self.sr_x_checks]==1]
        curr_lr_z_checks = self.lr_z_checks[self.route_confirmation_z[self.lr_z_checks]==1]
        curr_lr_x_checks = self.lr_x_checks[self.route_confirmation_x[self.lr_x_checks]==1]

        curr_z_checks = np.concatenate([curr_sr_z_checks, curr_lr_z_checks])
        curr_x_checks = np.concatenate([curr_sr_x_checks, curr_lr_x_checks])
        all_z_checks = np.concatenate([self.sr_z_checks, self.lr_z_checks])
        all_x_checks = np.concatenate([self.sr_x_checks, self.lr_x_checks])

        self.c += self.all_checks(self.p if with_gate_noise else 0, scale=True)

        if with_synd_noise: self.c.append("X_ERROR", [self.all_qbts[self.z_checks[z_check]] for z_check in curr_z_checks], self.p)
        if with_synd_noise: self.c.append("X_ERROR", [self.all_qbts[self.x_checks[x_check]] for x_check in curr_x_checks], self.p)

        for i, z_check in enumerate(curr_z_checks):
            self.c.append("M", self.all_qbts[self.z_checks[z_check]])
            self.curr_meas_z[z_check] = self.c.num_measurements
        for i, z_check in enumerate(all_z_checks):
            self.c.append("R", self.all_qbts[self.z_checks[z_check]])
        for i, x_check in enumerate(curr_x_checks):
            self.c.append("M", self.all_qbts[self.x_checks[x_check]])
            self.curr_meas_x[x_check] = self.c.num_measurements
        for i, x_check in enumerate(all_x_checks):
            self.c.append("R", self.all_qbts[self.x_checks[x_check]])

        if with_synd_noise: self.c.append("X_ERROR", [self.all_qbts[self.z_checks[z_check]] for z_check in all_z_checks], self.p/10)
        if with_synd_noise: self.c.append("X_ERROR", [self.all_qbts[self.x_checks[x_check]] for x_check in all_x_checks], self.p/10)

    def simulate(self, Theta, Pauli, only_coherent_error=False):
        # for i in range(1,self.num_rounds):
        self.c.append("SHIFT_COORDS", [], (0,0,1))
        # if (i%self.lr_time==0):
        if only_coherent_error ==False:
            self.c.append("DEPOLARIZE1", [self.all_qbts[qbt] for qbt in self.qbts], 0.001+2*(self.code[12]+self.code[13])*self.idle_error)
        else:# coherent error
            self.c.append("Z_ERROR",[self.all_qbts[qbt] for qbt in self.qbts[::-1]], np.sin(self.p)**2)
            # self.c += generate_error(Theta, Pauli)
        self.route_confirmation_z[self.sr_z_checks] = [1 for z in self.sr_z_checks]#[1 if np.random.random() < z_check_succ_probs[z] else 0 for z in sr_z_checks]
        self.route_confirmation_z[self.lr_z_checks] = [1 for z in self.lr_z_checks]#[1 if np.random.random() < z_check_succ_probs[z] else 0 for z in lr_z_checks]
        self.route_confirmation_x[self.sr_x_checks] = [1 for x in self.sr_x_checks]#[1 if np.random.random() < x_check_succ_probs[x] else 0 for x in sr_x_checks]
        self.route_confirmation_x[self.lr_x_checks] = [1 for x in self.lr_x_checks]#[1 if np.random.random() < x_check_succ_probs[x] else 0 for x in lr_x_checks]
        self.detector_history = np.vstack([self.detector_history, self.route_confirmation_z])
        self.lr_round(only_coherent_error)
            # else:
            #     self.c.append("DEPOLARIZE1", [all_qbts[qbt] for qbt in qbts], 0.001+2*code[12]*idle_error)
            #     self.route_confirmation_z[sr_z_checks] = [1 for z in sr_z_checks] #[1 if np.random.random() < z_check_succ_probs[z] else 0 for z in sr_z_checks]
            #     self.route_confirmation_z[lr_z_checks] = [1 for z in lr_z_checks]#0
            #     self.route_confirmation_x[sr_x_checks] = [1 for z in sr_x_checks]#[1 if np.random.random() < x_check_succ_probs[x] else 0 for x in sr_x_checks]
            #     self.route_confirmation_x[lr_x_checks] = [1 for z in lr_x_checks]#0
            #     self.detector_history = np.vstack([self.detector_history, self.route_confirmation_z])
            #     self.sr_round()
        self.detectors()

        # self.route_confirmation_z = np.ones(m*self.ell)
        # self.route_confirmation_x = np.ones(m*self.ell)
        # self.detector_history = np.vstack([self.detector_history, self.route_confirmation_z])
        # self.lr_round(with_gate_noise=False, with_synd_noise=False)
        # self.detectors(False)

        self.c.append("M",[self.all_qbts[qbt] for qbt in self.qbts[::-1]])
        self.observables(False)
