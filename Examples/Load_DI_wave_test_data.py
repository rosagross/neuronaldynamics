import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

current_directory = os.path.dirname(__file__)
path_up = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Model'))

########################################################################################################################
# DATA PREPROCESSED BY VINCENT
########################################################################################################################
data_fname = os.path.join(path_up, 's2020_043_CNS2023.mat')
data = scipy.io.loadmat(data_fname)
mean_DI_waves_detrend = data['meanDIwaves_detrend']
mean_DI_waves = data['meanDIwaves']
t = np.array(data['times'])[0]
t_new = np.arange(0, 12, 0.01)
di_data_detrend = np.interp(t_new, t, mean_DI_waves_detrend[:, 0])
di_data_mean = np.interp(t_new, t, mean_DI_waves[:, 0])
# plt.plot(t_new, di_data_detrend)
# plt.plot(t_new, di_data_mean)
# plt.ylim(-10, 10)
# plt.show()

########################################################################################################################
# DATA PREPROCESSED BY VINCENT
########################################################################################################################
nextcloud_path = 'C:\\Users\\emueller\\nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\DIwaves_Di_Lazzaro'
data_fname = os.path.join(nextcloud_path, 's2020_043_3ch.mat')
data = scipy.io.loadmat(data_fname)
###################
# f000: PA 120% RMT
###################

# channel 1 EMG
channel_1_data = data['f000_wave_data'][0][0][9].T[:, 0, :]
t_ch1 = np.linspace(0, channel_1_data[0].shape[0]/1e4, channel_1_data[0].shape[0]) * 1e2
# channel 2 & 3: EPIDURAL
channel_2_data = data['f000_wave_data'][0][0][9].T[:, 1, :]
channel_3_data = data['f000_wave_data'][0][0][9].T[:, 2, :]
t_ch2 = np.linspace(0, channel_2_data[0].shape[0]/1e4, channel_2_data[0].shape[0]) * 1e2
t_ch3 = np.linspace(0, channel_3_data[0].shape[0]/1e4, channel_3_data[0].shape[0]) * 1e2

mean_ch1 = channel_1_data.mean(axis=0)
mean_ch2 = channel_2_data.mean(axis=0)
mean_ch3 = channel_3_data.mean(axis=0)

tms_peak_idx, tms_peaks = scipy.signal.find_peaks(mean_ch2, height=200)
t_peak = t_ch1[tms_peak_idx]
t_ch1 -= t_peak
for i in range(channel_1_data.shape[0]):
    plt.plot(t_ch1, channel_1_data[i], c='k', alpha=0.2)
plt.plot(t_ch1, mean_ch1, c='blue')
plt.ylim(-0.5, 0.5)
plt.xlabel('t (ms)')
plt.ylabel('EMG (mV)')
plt.show()

for i in range(channel_2_data.shape[0]):
    plt.plot(t_ch1, channel_2_data[i], c='k', alpha=0.2)
plt.plot(t_ch1, mean_ch2, c='blue')
plt.ylim(-10, 10)
plt.xlabel('t (ms)')
plt.ylabel('Epidural potential (µV)')
plt.show()

for i in range(channel_3_data.shape[0]):
    plt.plot(t_ch1, channel_3_data[i], c='k', alpha=0.2)
plt.plot(t_ch1, mean_ch3, c='blue')
plt.ylim(-10, 10)
plt.xlabel('t (ms)')
plt.ylabel('Epidural potential (µV)')
plt.show()
