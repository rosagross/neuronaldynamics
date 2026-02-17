import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Utils import plot_DI_wave_data, get_di_wave_data
matplotlib.use('TkAgg')
# nextcloud_path = 'C:\\Users\\emueller\\nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\DIwaves_Di_Lazzaro'
nextcloud_path = 'C:\\Users\\User\\nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\DIwaves_Di_Lazzaro'
nextcloud_path = '/home/erik/Downloads/DI_wave_data/DIwaves_Di_Lazzaro'
plot = True
detrend = False


def add_entry(dict_target, dict_source, orientation, threshold_type, rmt_digit, recording, rmt_value, year, name):
    if isinstance(rmt_digit, list):
        rmt_digit = rmt_digit[0]
    # init dict if necessary
    if not orientation in dict_target.keys():
        dict_target[orientation] = {}
    if not threshold_type in dict_target[orientation].keys():
        dict_target[orientation][threshold_type] = {}
    if not str(rmt_digit) in dict_target[orientation][threshold_type].keys():
        dict_target[orientation][threshold_type][str(rmt_digit)] = {}
    if not year in dict_target[orientation][threshold_type][str(rmt_digit)].keys():
        dict_target[orientation][threshold_type][str(rmt_digit)][year] = {}

    dict_target[orientation][threshold_type][str(rmt_digit)][year][name] = {}
    channel_names = dict_source[recording][rmt_value].keys()
    dict_target[orientation][threshold_type][str(rmt_digit)][year][name] = \
        dict(time=dict_source[recording][rmt_value]['time'],
             time_short=dict_source[recording][rmt_value]['time_short'],

             emg_location=dict_source[recording]['emg_location'])
    for channel in channel_names:
        signal = dict_source[recording][rmt_value]['signal'],
        signal_short = dict_source[recording][rmt_value]['signal_short'],


di_wave_data_collection = dict()
########################################################################################################################
# Di Lazarro DATA 2020 3ch - PA
########################################################################################################################

data_fname = os.path.join(nextcloud_path, 's2020_043_3ch.mat')
data_2020 = scipy.io.loadmat(data_fname)
rmt_names = ['f002_wave_data', 'f000_wave_data', 'f001_wave_data']
titles = ['Ch1: EMG (mV)', 'Ch3: Epidural potential (µV)', 'Ch4: Epidural potential (µV)']
yaxis = ['100% RMT PA', '120% RMT PA', '140% RMT PA']
alphas = [0.2, 0.05, 0.05]
title = 'Di Lazarro 2020 PA'
if plot:
    plot_DI_wave_data(data_2020, rmt_names, titles, yaxis, alphas, filter=True, main_title=title, do_detrend=detrend)

########################################################################################################################
# Di Lazarro DATA 2020 3ch - LM
########################################################################################################################

rmt_names_2 = ['f013_wave_data', 'f011_wave_data', 'f012_wave_data']
yaxis_2 = ['80% RMT LM', '100% RMT LM', '120% RMT LM']
alphas_2 = [0.2, 0.05, 0.05]
title_2 = 'Di Lazarro 2020 LM'
if plot:
    plot_DI_wave_data(data_2020, rmt_names_2, titles, yaxis_2, alphas_2, filter=True, main_title=title_2, do_detrend=detrend)

########################################################################################################################
# Di Lazarro DATA 2013 3ch
########################################################################################################################

data_fname = os.path.join(nextcloud_path, 's2013_031_3ch.mat')
data_2013 = scipy.io.loadmat(data_fname)
rmt_names_3 = ['f002_wave_data', 'f001_wave_data', 'f002_wave_data']
titles_3 = ['Ch1: EMG (mV)', 'Ch2: Epidural potential (µV)', 'Ch3: Epidural potential (µV)']
yaxis_3 = ['100% RMT - LM', '110% RMT - PA', '120% RMT - PA']
alphas_3 = [0.2, 0.05, 0.05]
title_3 = 'Di Lazarro 2013 PA & LM'
if plot:
    plot_DI_wave_data(data_2013, rmt_names_3, titles_3, yaxis_3, alphas_3, filter=True, main_title=title_3,
                      do_detrend=detrend)

########################################################################################################################
# Di Lazarro DATA 2007 2ch
########################################################################################################################

data_fname = os.path.join(nextcloud_path, 's2007_epid001.mat')
data_2007 = scipy.io.loadmat(data_fname)
rmt_names_4 = ['f001_wave_data', 'f002_wave_data', 'f003_wave_data']
titles_4 = ['Ch1: EMG (mV)', 'Ch2: Epidural potential (µV)']
yaxis_4 = ['120% RMT - PA', '150% RMT - PA', '120% RMT - LM']
alphas_4 = [0.2, 0.2, 0.2]
title_4 = 'Di Lazarro 2007 PA & LM'
if plot:
    plot_DI_wave_data(data_2007, rmt_names_4, titles_4, yaxis_4, alphas_4, filter=True, main_title=title_4, n_channels=2,
                      emg_peak_height=20, do_detrend=detrend, find_peaks_args=dict(threshold=0.01, distance=1))

########################################################################################################################
# Di Lazarro DATA 2004 2ch #1
########################################################################################################################

data_fname = os.path.join(nextcloud_path, 's2004_epid001.mat')
data_2004_1 = scipy.io.loadmat(data_fname)
rmt_names_5 = ['f001_wave_data', 'f002_wave_data', 'f003_wave_data', 'f003_wave_data']
titles_5 = ['Ch1: EMG (mV)', 'Ch2: Epidural potential (µV)']
yaxis_5 = ['PA 150% RMT', 'PA 155% RMT', 'LM 140% RMT', 'LM 140% RMT']
alphas_5 = [0.3, 0.1, 0.1]
title_5 = 'Di Lazarro 2004 PA & LM # 1'
if plot:
    plot_DI_wave_data(data_2004_1, rmt_names_5, titles_5, yaxis_5, alphas_5, filter=True, main_title=title_5, n_channels=2,
                      emg_peak_height=200, sample_frequency=[5e3, 25e3, 5e3, 5e3], do_detrend=detrend,
                      switch_channel_order=True, find_peaks_args=dict(threshold=0.005, distance=1))

########################################################################################################################
# Di Lazarro DATA 2004 2ch #2
########################################################################################################################

data_fname = os.path.join(nextcloud_path, 's2004_epid002.mat')
data_2004_2 = scipy.io.loadmat(data_fname)
rmt_names_6 = ['f001_wave_data', 'f002_wave_data', 'f003_wave_data']
titles_6 = ['Ch1: EMG (mV)', 'Ch2: Epidural potential (µV)']
yaxis_6 = ['LM 50% MSO', 'PA 154% RMT', 'PA 146% RMT']
alphas_6 = [0.3, 0.1, 0.1]
title_6 = 'Di Lazarro 2004 PA & LM # 2'
if plot:
    plot_DI_wave_data(data_2004_2, rmt_names_6, titles_6, yaxis_6, alphas_6, filter=True, main_title=title_6, n_channels=2,
                      emg_peak_height=40, sample_frequency=5e3, do_detrend=True, switch_channel_order=True,
                      find_peaks_args=dict(threshold=0.005, distance=1))


meta_data_2004_1 = dict(channel_names=titles_5, emg_location='left FDI muscle', name=title_5, year='2004')
meta_data_2004_2 = dict(channel_names=titles_6, emg_location='left FDI muscle', name=title_6, year='2004')
meta_data_2007 = dict(channel_names=titles_4, emg_location='left FDI muscle', name=title_4, year='2007')
meta_data_2013 = dict(channel_names=titles_3, emg_location='left APB muscle', name=title_3, year='20013')
meta_data_2020_PA = dict(channel_names=titles, emg_location='left FDI muscle', name=title, year='2020')
meta_data_2020_LM = dict(channel_names=titles, emg_location='left FDI muscle', name=title_2, year='2020')

di_wave_data_collection['s2004_epid001'] = get_di_wave_data(data_2004_1, rmt_names_5, epidural_channel_idxs=[0],
                                                            find_peaks_args=dict(threshold=0.005, distance=1),
                                                            meta_data=meta_data_2004_1,
                                                            sample_frequency=[5e3, 25e3, 5e3, 5e3],
                                                            rmt_values=yaxis_5)
# TODO: sth wrong with this channel here, I believe it does find the epidural channel instead of the EMG
#  for the peak detection
di_wave_data_collection['s2004_epid002'] = get_di_wave_data(data_2004_2, rmt_names_6, epidural_channel_idxs=[0],
                                                            find_peaks_args=dict(threshold=0.005, distance=1),
                                                            meta_data=meta_data_2004_2,
                                                            emg_peak_height=1.5,
                                                            sample_frequency=5e3,
                                                            rmt_values=yaxis_6)
di_wave_data_collection['s2007_epid001'] = get_di_wave_data(data_2007, rmt_names_4, epidural_channel_idxs=[1],
                                                            emg_peak_height=0.4, find_peaks_args=dict(threshold=0.01, distance=1),
                                                            meta_data=meta_data_2007, rmt_values=yaxis_4)
di_wave_data_collection['s2013_031_3ch'] = get_di_wave_data(data_2013, rmt_names_3, epidural_channel_idxs=[1, 2],
                                                            meta_data=meta_data_2013, rmt_values=yaxis_3,
                                                            emg_peak_height=0.3)
di_wave_data_collection['s2020_043_3ch_PA'] = get_di_wave_data(data_2020, rmt_names, epidural_channel_idxs=[1, 2],
                                                               meta_data=meta_data_2020_PA, rmt_values=yaxis,
                                                               emg_peak_height=1.0)
di_wave_data_collection['s2020_043_3ch_LM'] = get_di_wave_data(data_2020, rmt_names_2, epidural_channel_idxs=[1, 2],
                                                               meta_data=meta_data_2020_LM, rmt_values=yaxis_2,
                                                               emg_peak_height=2.0)

# TODO: structure LM/PM - threshold - year - subject_ID - Channel
# TODO: structure from file: year + subject_ID - channel -
#  [data_DI-waves, data_DI-waves_short, threshold_type, threshold, orientation, time, time_short, data_name]
# bad channels 2004_2, 2020 - PA/LM - Ch4

# create newly structured hdf5
#
di_wave_dict = {}
for recording in di_wave_data_collection.keys():
    name = di_wave_data_collection[recording]['name']
    year = di_wave_data_collection[recording]['year']
    emg_location = di_wave_data_collection[recording]['emg_location']
    for rmt_value in di_wave_data_collection[recording].keys():
        if rmt_value not in ['channel_names', 'emg_location', 'name', 'year']:  # try to exclude non_data keys
            orientation = di_wave_data_collection[recording][rmt_value]['orientation']
            threshold_type = di_wave_data_collection[recording][rmt_value]['threshold_type']
            rmt_digit = di_wave_data_collection[recording][rmt_value]['RMT_digit']
            add_entry(dict_target=di_wave_dict,
                      dict_source=di_wave_data_collection,
                      orientation=orientation,
                      threshold_type=threshold_type,
                      rmt_digit=rmt_digit,
                      recording=recording,
                      rmt_value=rmt_value,
                      year=year,
                      name=name)



a=1
# for k, v in d.items():
#     h.create_dataset(k, data=np.array(v))