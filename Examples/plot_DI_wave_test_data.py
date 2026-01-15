import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Utils import butter_highpass_filter, detrend
matplotlib.use('TkAgg')



########################################################################################################################
# UNPROCESSED DATA 2020 3ch
########################################################################################################################
nextcloud_path = 'C:\\Users\\emueller\\nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\DIwaves_Di_Lazzaro'
# nextcloud_path = 'C:\\Users\\User\\nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\DIwaves_Di_Lazzaro'

data_fname = os.path.join(nextcloud_path, 's2020_043_3ch.mat')
data = scipy.io.loadmat(data_fname)
##############################################################################
# f002: PA 100% RMT, f000: PA 120% RMT, f001: PA 140% RMT
##############################################################################
rmt_names = ['f002_wave_data', 'f000_wave_data', 'f001_wave_data']
titles = ['Ch1: EMG (mV)', 'Ch3: Epidural potential (µV)', 'Ch4: Epidural potential (µV)']
yaxis = ['100% RMT', '120% RMT', '140% RMT']
alphas = [0.2, 0.05, 0.05]

def plot_data(data, rmt_names, titles, yaxis, alphas, filter=True, do_detrend=True):
    fig = plt.figure(figsize=(12, 8))
    for k in range(3):
        rmt_k_data = data[rmt_names[k]][0][0][9].T
        # channel 1 EMG, channel 2 & 3: EPIDURAL
        t_ch1 = np.linspace(0, rmt_k_data.shape[2]/1e4, rmt_k_data.shape[2]) * 1e3
        mean_ch2 = rmt_k_data[:, 1, :].mean(axis=0)
        tms_peak_idx, tms_peaks = scipy.signal.find_peaks(mean_ch2, height=200)
        t_peak = t_ch1[tms_peak_idx]
        t_ch1 -= t_peak
        dt = np.diff(t_ch1)[0]
        t_1ms = int(t_peak + 1/dt)
        for j in range(3):
            ax = fig.add_subplot(3, 3, (3*k)+j+1)
            for i in range(rmt_k_data.shape[0]):
                if j > 0 and do_detrend:
                    data_i = detrend(t_ch1, rmt_k_data[i, j], mean_cutoff_idx=t_1ms)
                else:
                    data_i = rmt_k_data[i, j]
                plt.plot(t_ch1, data_i, c='k', alpha=alphas[j])
            ax_mean = rmt_k_data[:, j].mean(axis=0)
            # test gaussian filter, highpass filter, centering to zero

            if j > 0:
                if do_detrend:
                    ax_mean = detrend(t_ch1, ax_mean, mean_cutoff_idx=t_1ms)
                # ax_mean_filtered = butter_highpass_filter(ax_mean, cutoff=0.1, fps=int(1 / dt))
                if filter:
                    ax_mean = scipy.ndimage.gaussian_filter1d(ax_mean, sigma=1)
                ax.plot(t_ch1, ax_mean, c='blue')
                ax.set_xticks(np.array([0, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10]))
                ax.set_xticklabels(['0', '2', ' ', '3', ' ', '4', ' ', '5', ' ', '6', ' ','7', ' ', '8', ' ', '9', ' ','10'])
            else:
                ax.plot(t_ch1, ax_mean, c='blue')
                ax.set_xticks(np.array([-2, 0, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])*10)
                ax.set_xticklabels(['-20', '0', '20', ' ', '30', ' ', '40', ' ', '50', ' ', '60', ' '])
            # ax.tick_params(axis='x', which='major', labelsize=7)
            if j < 1:
                if k < 1:
                    ax.set_ylim(-0.1, 0.1)
                else:
                    ax.set_ylim(-0.5, 0.5)
                ax.set_ylabel(yaxis[k], fontsize=12)
            else:
                if do_detrend:
                    ax.set_ylim(-0.5, 5)
                else:
                    ax.set_ylim(-5, 8)
                ax.set_xlim(0, 12)
                ax.vlines([5], -10, 10, color='k')
            if k < 1:
                ax.set_title(titles[j])
            if k > 1:
                ax.set_xlabel('t (ms)')
            ax.grid()

    plt.show()

plot_data(data, rmt_names, titles, yaxis, alphas, filter=True)
rmt_names_2 = ['f013_wave_data', 'f011_wave_data', 'f012_wave_data']
yaxis_2 = ['80% RMT', '100% RMT', '120% RMT']
alphas_2 = [0.2, 0.05, 0.05]
plot_data(data, rmt_names_2, titles, yaxis_2, alphas_2, filter=True)

data_fname = os.path.join(nextcloud_path, 's2013_031_3ch.mat')
data_2013 = scipy.io.loadmat(data_fname)
rmt_names_3 = ['f002_wave_data', 'f001_wave_data', 'f000_wave_data']
titles_3 = ['Ch1: EMG (mV)', 'Ch2: Epidural potential (µV)', 'Ch3: Epidural potential (µV)']
yaxis_3 = ['100% RMT - LM', '110% RMT - PA', '120% RMT - PA']
alphas_3 = [0.2, 0.05, 0.05]
plot_data(data, rmt_names_3, titles_3, yaxis_3, alphas_3, filter=True)


l = 12