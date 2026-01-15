import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Utils import plot_DI_wave_data
matplotlib.use('TkAgg')
nextcloud_path = 'C:\\Users\\emueller\\nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\DIwaves_Di_Lazzaro'
# nextcloud_path = 'C:\\Users\\User\\nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\DIwaves_Di_Lazzaro'

########################################################################################################################
# Di Lazarro DATA 2007 2ch
########################################################################################################################

data_fname = os.path.join(nextcloud_path, 's2007_epid001.mat')
data_2007 = scipy.io.loadmat(data_fname)
rmt_names_4 = ['f001_wave_data', 'f002_wave_data', 'f003_wave_data']
titles_4 = ['Ch1: EMG (mV)', 'Ch2: Epidural potential (µV)']
yaxis_4 = ['120% RMT - PA', '150% RMT - PA', '120% RMT - LM']
alphas_4 = [0.2, 0.05, 0.05]
title_4 = 'Di Lazarro 2007 PA & LM'
plot_DI_wave_data(data_2007, rmt_names_4, titles_4, yaxis_4, alphas_4, filter=True, main_title=title_4, n_channels=2,
                  emg_peak_height=20)

########################################################################################################################
# Di Lazarro DATA 2020 3ch - PA
########################################################################################################################

data_fname = os.path.join(nextcloud_path, 's2020_043_3ch.mat')
data = scipy.io.loadmat(data_fname)
rmt_names = ['f002_wave_data', 'f000_wave_data', 'f001_wave_data']
titles = ['Ch1: EMG (mV)', 'Ch3: Epidural potential (µV)', 'Ch4: Epidural potential (µV)']
yaxis = ['100% RMT', '120% RMT', '140% RMT']
alphas = [0.2, 0.05, 0.05]
title = 'Di Lazarro 2020 PA'
plot_DI_wave_data(data, rmt_names, titles, yaxis, alphas, filter=True, main_title=title)

########################################################################################################################
# Di Lazarro DATA 2020 3ch - LM
########################################################################################################################

rmt_names_2 = ['f013_wave_data', 'f011_wave_data', 'f012_wave_data']
yaxis_2 = ['80% RMT', '100% RMT', '120% RMT']
alphas_2 = [0.2, 0.05, 0.05]
title_2 = 'Di Lazarro 2020 LM'
plot_DI_wave_data(data, rmt_names_2, titles, yaxis_2, alphas_2, filter=True, main_title=title_2)

########################################################################################################################
# Di Lazarro DATA 2013 3ch
########################################################################################################################

data_fname = os.path.join(nextcloud_path, 's2013_031_3ch.mat')
data_2013 = scipy.io.loadmat(data_fname)
rmt_names_3 = ['f002_wave_data', 'f000_wave_data', 'f001_wave_data']
# TODO: major difference in naming here should be ['f002_wave_data', 'f001_wave_data', 'f000_wave_data']
#  for correct thresholds! Was it save wrong?
titles_3 = ['Ch1: EMG (mV)', 'Ch2: Epidural potential (µV)', 'Ch3: Epidural potential (µV)']
yaxis_3 = ['100% RMT - LM', '110% RMT - PA', '120% RMT - PA']
alphas_3 = [0.2, 0.05, 0.05]
title_3 = 'Di Lazarro 2013 PA & LM'
plot_DI_wave_data(data_2013, rmt_names_3, titles_3, yaxis_3, alphas_3, filter=True, main_title=title_3)




########################################################################################################################
# Di Lazarro DATA 2004 2ch #1
########################################################################################################################

########################################################################################################################
# Di Lazarro DATA 2004 2ch #2
########################################################################################################################