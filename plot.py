import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns

def plot_estimation(data1, data2, data3, data4, true_value, first_value=0, second_value=1, model_name='xx'):
    stderr1 = np.mean((data1 - true_value)**2, axis=0)
    stderr2 = np.mean((data2 - true_value)**2, axis=0)
    stderr3 = np.mean((data3 - true_value)**2, axis=0)
    stderr4 = np.mean((data4 - true_value)**2, axis=0)
    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)
    mean3 = np.mean(data3, axis=0)
    mean4 = np.mean(data4, axis=0)
    plot_index = np.arange(11)[1:]
    fig = plt.figure()
    axx = fig.add_axes([0.15,0.7,0.7,0.25])
    axx.errorbar(plot_index, mean1[plot_index], marker='v', ls='--', yerr=stderr1[plot_index])
    axx.errorbar(plot_index, mean2[plot_index], marker='x', ls='--', yerr=stderr2[plot_index])
    axx.errorbar(plot_index, mean3[plot_index], marker='*', ls='--', yerr=stderr3[plot_index])
    axx.errorbar(plot_index, mean4[plot_index], marker='d', ls='--', yerr=stderr4[plot_index])
    # axx.plot(plot_index, mean1[plot_index], ls='--')
    # axx.plot(plot_index, mean2[plot_index], ls='-.')
    # axx.plot(plot_index, mean3[plot_index], ls=':')
    # axx.plot(plot_index, mean4[plot_index], ls='-')
    axx.plot(plot_index, np.ones(len(plot_index))*true_value, '-', color='black')
    axx.set_title('(a)')
    axx.set_xlim(0, 10.5)
    # axx.grid(True)

    # axx.get_xaxis().set_visible(False)
    ax = fig.add_axes([0.15,0.1,0.7,0.5])
    ax.set_yscale('log')
    if model_name == 'magnetometer':
        ax.plot(plot_index, stderr1[plot_index], marker='v', ls='--', label=r'Correlated $(\hat{\theta}_0=$'+'{})'.format(first_value))
        ax.plot(plot_index, stderr2[plot_index], marker='x', ls='--', label=r'Unitary $(\hat{\theta}_0=$'+'{})'.format(first_value))
        # ax.plot(plot_index, stderr3[plot_index], marker='*', ls='--', label=r'Correlated $(\hat{\theta}=$'+'{})'.format(second_value))
        # ax.plot(plot_index, stderr4[plot_index], marker='d', ls='--', label=r'Unitary $(\hat{\theta}=$'+'{})'.format(second_value))
        ax.plot(plot_index, stderr3[plot_index], marker='*', ls='--', label=r'Correlated $(\hat{\theta}_0=\frac{\pi}{2})$')
        ax.plot(plot_index, stderr4[plot_index], marker='d', ls='--', label=r'Unitary $(\hat{\theta}_0=\frac{\pi}{2})$')
        ax.set_ylabel(r'$\mathbb{E}[(\theta-\hat{\theta})^2]$', fontsize=15)
        axx.set_ylabel(r'$\hat{\theta}$', fontsize=15)
        qcrb_unitary = 1 / (4 * 100 * 0.1**2 * 5**2)
        qcrb_correlated = 1 / (100 * (4 * 0.1**2 * 5**2 + 4 * 0.05 * 5))
        ax.plot(np.arange(11), np.ones(11) * qcrb_unitary, ls='-.', label='QCRB-Unitary')
        ax.plot(np.arange(11), np.ones(11) * qcrb_correlated, ls=':', label='QCRB-Correlated')
        ax.set_yticks([0.1, 0.01, qcrb_correlated])
        ax.set_yticklabels([r'$10^{-1}$', r'$10^{-2}$',r'$5 \times 10^{-3}$'])
        ax.set_xlim(0, 10.5)
        plt.legend(fontsize=10, loc=(0.65,0.5))
    elif model_name == 'gyroscope':
        ax.plot(plot_index, stderr1[plot_index], marker='v', ls='--', label=r'Correlated $(\hat{\Omega}_0$'+'={})'.format(first_value))
        ax.plot(plot_index, stderr2[plot_index], marker='x', ls='--', label=r'Unitary $(\hat{\Omega}_0=$' +'{})'.format(first_value))
        ax.plot(plot_index, stderr3[plot_index], marker='*', ls='--', label=r'Correlated $(\hat{\Omega}_0=$'+'{})'.format(second_value))
        ax.plot(plot_index, stderr4[plot_index], marker='d', ls='--', label=r'Unitary $(\hat{\Omega}_0=$'+'{})'.format(second_value))
        ax.set_ylabel(r'$\mathbb{E}[(\Omega-\hat{\Omega})^2]$', fontsize=15)
        axx.set_ylabel(r'$\hat{\Omega}$', fontsize=15)
        qcrb_unitary = 0.01 / (0.1**2 * 5**4)
        qcrb_correlated = 0.01 / (0.1**2 * 5**4 + 4/3 * 0.05 * 5**3)
        ax.plot(np.arange(11), np.ones(11) * qcrb_unitary, ls='-.', label='QCRB-Unitary')
        ax.plot(np.arange(11), np.ones(11) * qcrb_correlated, ls=':', label='QCRB-Correlated')
        ax.set_yticks([0.01,qcrb_unitary, qcrb_correlated])
        ax.set_yticklabels([r'$10^{-2}$',r'$1.6\times 10^{-3}$', r'$6.9\times 10^{-4}$'])
        ax.set_xlim(0,10.5)
        plt.legend(fontsize=10, loc=(0.65, 0.8))
    # ax.grid(True)
    ax.set_title('(b)')
    ax.set_xlabel('adaptive steps',fontsize=15)


    plt.show()


 


if __name__ == '__main__':
    data1 = np.load('./data/GyroscopeQEC02t5.npy')
    data2 = np.load('./data/GyroscopeUnitary02t5.npy')
    data3 = np.load('./data/GyroscopeQEC04t5.npy')
    data4 = np.load('./data/GyroscopeUnitary04t5.npy')
    true_value = 0.3
    first_value = 0.2
    second_value = 0.4
    model_name = 'gyroscope'
    data1 = np.load('./data/MagnetometerQEC.npy')
    data2 = np.load('./data/MagnetometerUnitary.npy')
    data3 = np.load('./data/MagnetometerQECpi_2.npy')
    data4 = np.load('./data/MagnetometerUnitarypi_2.npy')
    true_value = np.pi/4
    first_value = 0
    second_value = np.pi/2
    model_name = 'magnetometer'
    plot_estimation(data1, data2, data3, data4, true_value=true_value, first_value=first_value, second_value=second_value, model_name=model_name)