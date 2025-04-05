import numpy as np
import datetime
import matplotlib.pyplot as plt

import prediction_random_sample

def main():
    d = 3
    K = 15
    N_presample = 10**3
    sample_size = 10**2
    N = 10**3
    Pauli = [('Z', i) for i in range((d+1)**2)]
    
    theta_list = np.linspace(0, 0.3, 100)
    
    F_K_list = []
    SD_K_list = []
    F_theta = []
    SD_theta = []
    for theta in theta_list:
        Theta = np.array([theta for _ in range(len(Pauli))])
        F_K, SD_K = prediction_random_sample.F_random_sample_main(d, Pauli, K, Theta, N, sample_size, theta, 'pre-sample', N_presample)
        F_K_list.append(F_K)
        SD_K_list.append(SD_K)
        F_theta.append(sum(F_K))
        SD_theta.append(sum(SD_K))
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')    
    file_name = 'template/data/' + current_time
    np.savez(file_name, F_K_list=F_K_list, SD_K_list=SD_K_list, theta_list=theta_list, d=d, K=K, N_presample=N_presample, sample_size=sample_size, N=N)
    
    # plot
    plt.figure(dpi=300)
    plt.scatter(theta_list, F_theta, label='F')
    plt.plot(theta_list, F_theta)
    plt.xlabel('theta')
    plt.ylabel('F')
    plt.legend()
    file_name_ = 'template/image/' + current_time
    plt.savefig(file_name_ + '_F.png')
    
    plt.figure(dpi=300)
    plt.scatter(theta_list, SD_theta, label='SD')
    plt.plot(theta_list, SD_theta)
    plt.xlabel('theta')
    plt.ylabel('SD')
    plt.legend()
    plt.savefig(file_name_ + '_SD.png')
    
    plt.figure(dpi=300)
    K_list = np.arange(1, K+1)
    theta_ = theta_list[int(len(theta_list)/2)]
    plt.scatter(K_list, F_K_list[int(len(theta_list)/2)], label='F')
    plt.plot(K_list, F_K_list[int(len(theta_list)/2)], label='F')
    plt.scatter(K_list, SD_K_list[int(len(theta_list)/2)], label='SD')
    plt.plot(K_list, SD_K_list[int(len(theta_list)/2)], label='SD')
    plt.xlabel('K')
    plt.ylabel('F/SD')
    plt.legend()
    plt.savefig(file_name_ +'_theta:'+ str(theta_) + '_F_SD.png')
    
    
if __name__ == '__main__':
    time = datetime.datetime.now()
    main()
    time = datetime.datetime.now() - time
    print(time)