import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


########################################################################################################################
# UNPROCESSED DATA 2020 3ch
########################################################################################################################
nextcloud_path = 'C:\\Users\\emueller\\nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\DIwaves_Di_Lazzaro'
data_fname = os.path.join(nextcloud_path, 's2020_043_3ch.mat')
data = scipy.io.loadmat(data_fname)
##############################################################################
# f002: PA 100% RMT, f000: PA 120% RMT, f001: PA 140% RMT
##############################################################################
rmt_names = ['f002_wave_data', 'f000_wave_data', 'f001_wave_data']
titles = ['Ch1: EMG (mV)', 'Ch3: Epidural potential (µV)', 'Ch4: Epidural potential (µV)']
yaxis = ['100% RMT', '120% RMT', '140% RMT']

fig = plt.figure(figsize=(12, 8))
for k in range(3):
    rmt_k_data = data[rmt_names[k]][0][0][9].T
    # channel 1 EMG, channel 2 & 3: EPIDURAL
    t_ch1 = np.linspace(0, rmt_k_data.shape[2]/1e4, rmt_k_data.shape[2]) * 1e2
    mean_ch2 = rmt_k_data[:, 1, :].mean(axis=0)
    tms_peak_idx, tms_peaks = scipy.signal.find_peaks(mean_ch2, height=200)
    t_peak = t_ch1[tms_peak_idx]
    t_ch1 -= t_peak
    for j in range(3):
        ax = fig.add_subplot(3, 3, (3*k)+j+1)
        for i in range(rmt_k_data.shape[0]):
            plt.plot(t_ch1, rmt_k_data[i, j], c='k', alpha=0.2)
        ax_mean = rmt_k_data[:, j].mean(axis=0)
        ax_mean_filtered = scipy.ndimage.gaussian_filter1d(ax_mean, sigma=5)
        ax.plot(t_ch1, ax_mean_filtered, c='blue')
        if j < 1:
            if k < 1:
                ax.set_ylim(-0.1, 0.1)
            else:
                ax.set_ylim(-0.5, 0.5)
            ax.set_ylabel(yaxis[k], fontsize=12)
        else:
            ax.set_ylim(-5, 5)
        if k < 1:
            ax.set_title(titles[j])
        if k > 1:
            ax.set_xlabel('t (ms)')
        ax.grid()


plt.show()
