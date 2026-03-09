import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
from Model.Neck import generate_EP
import scipy
from Utils import get_peak_values, argmin_2d, butter_highpass_filter
import h5py
from tqdm.contrib import itertools
from Model.Nykamp_Model import Nykamp_Model_1
matplotlib.use('TkAgg')

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]
detrend = True
scale = False
plot_overview = True
opt_crit = 0

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
# hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
if not detrend:
    hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
    # hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
else:
    hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data_detrended.hdf5"
    # hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data_detrended.hdf5"


# hdf5_path = "/home/erik/Nextcloud_Uni/TMS Neuro Projects/M1_modeling/DI_wave_data/extracted_DI_waves/DiLazarro_di_wave_data.hdf5"
simulation_name = 'I_wave_plot'


measurement_dict_2020_140_PA_ch3 = dict(orientation='PA', threshold=140, year=2020, hdf5_path=hdf5_path, sigma=1.0)
measurement_dict_2020_120_PA_ch3 = dict(orientation='PA', threshold=120, year=2020, hdf5_path=hdf5_path, sigma=1.0)
measurement_dict_2020_100_PA_ch3 = dict(orientation='PA', threshold=100, year=2020, hdf5_path=hdf5_path, sigma=1.0)
measurement_dict_2013_110_PA_ch2 = dict(orientation='PA', threshold=110, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=0)
measurement_dict_2013_110_PA_ch3 = dict(orientation='PA', threshold=110, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=1)
measurement_dict_2007_120_PA_ch3 = dict(orientation='PA', threshold=120, year=2007, hdf5_path=hdf5_path, sigma=1.0)
measurement_dict_2007_150_PA_ch3 = dict(orientation='PA', threshold=150, year=2007, hdf5_path=hdf5_path, sigma=1.0)
measurement_dict_2004_154_PA_2_ch2 = dict(orientation='PA', threshold=154, year=2004, hdf5_path=hdf5_path, sigma=1.0)
measurement_dict_2004_146_PA_2_ch2 = dict(orientation='PA', threshold=146, year=2004, hdf5_path=hdf5_path, sigma=1.0)
measurement_dict_2004_150_PA_1_ch2 = dict(orientation='PA', threshold=150, year=2004, hdf5_path=hdf5_path, sigma=0.1)

measurement_dict_2020_80_LM_ch3 = dict(orientation='LM', threshold=80, year=2020, hdf5_path=hdf5_path, sigma=1.0, channel=0)
measurement_dict_2020_80_LM_ch4 = dict(orientation='LM', threshold=80, year=2020, hdf5_path=hdf5_path, sigma=1.0, channel=1)
measurement_dict_2020_100_LM_ch3 = dict(orientation='LM', threshold=100, year=2020, hdf5_path=hdf5_path, sigma=1.0, channel=0)
measurement_dict_2020_100_LM_ch4 = dict(orientation='LM', threshold=100, year=2020, hdf5_path=hdf5_path, sigma=1.0, channel=1)
measurement_dict_2020_120_LM_ch3 = dict(orientation='LM', threshold=120, year=2020, hdf5_path=hdf5_path, sigma=1.0, channel=0)
measurement_dict_2020_120_LM_ch4 = dict(orientation='LM', threshold=120, year=2020, hdf5_path=hdf5_path, sigma=1.0, channel=1)
measurement_dict_2013_100_LM_ch2 = dict(orientation='LM', threshold=100, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=0)
measurement_dict_2013_100_LM_ch3 = dict(orientation='LM', threshold=100, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=1)
measurement_dict_2007_120_LM_ch2 = dict(orientation='LM', threshold=120, year=2007, hdf5_path=hdf5_path, sigma=1.0, channel=0)
measurement_dict_2004_140_LM_1_ch2 = dict(orientation='LM', threshold=140, year=2004, hdf5_path=hdf5_path, sigma=0.1, channel=0)

data_dicts = [measurement_dict_2020_140_PA_ch3,
              measurement_dict_2020_120_PA_ch3,
              measurement_dict_2020_100_PA_ch3,
              measurement_dict_2013_110_PA_ch2,
              measurement_dict_2013_110_PA_ch3,
              measurement_dict_2007_120_PA_ch3,
              measurement_dict_2007_150_PA_ch3,
              measurement_dict_2004_154_PA_2_ch2,
              measurement_dict_2004_146_PA_2_ch2,
              measurement_dict_2004_150_PA_1_ch2]

data_dicts_LM = [measurement_dict_2020_80_LM_ch3,
              measurement_dict_2020_80_LM_ch4,
              measurement_dict_2020_100_LM_ch3,
              measurement_dict_2020_100_LM_ch4,
              measurement_dict_2020_120_LM_ch3,
              measurement_dict_2020_120_LM_ch4,
              measurement_dict_2013_100_LM_ch2,
              measurement_dict_2013_100_LM_ch3,
              # measurement_dict_2007_120_LM_ch2,
              measurement_dict_2004_140_LM_1_ch2]

def load_recordings(load_dict):
    data_dict = dict(orientation='PA', threshold=100, year=2020, threshold_type='RMT', channel=0, subject=0,
                     hdf5_path=hdf5_path, sigma=1)
    data_dict.update(load_dict)
    with h5py.File(data_dict['hdf5_path'], 'r') as h5file:
        name_h5group = h5file[data_dict['orientation']][data_dict['threshold_type']][str(data_dict['threshold'])][
            str(data_dict['year'])]
        name_dict = dict(name_h5group)
        name_keys = name_dict.keys()

        di_signals = []
        times = []

        subjects = []
        subject = data_dict['subject']
        if not isinstance(subject, list):
            subject = [subject]
        for i_key, key in enumerate(name_keys):
            if i_key in subject:
                subject_i = name_h5group[key]

                subjects.append(subject_i)

                times.append(np.array(subject_i['time_short']))
                channel_h5subgroup = subject_i
                channel_keys = dict(subject_i).keys()
                single_channels = []
                channel = data_dict['channel']
                if not isinstance(channel, list):
                    channel = [channel]
                for i_key, key in enumerate(channel_keys):
                    if i_key in channel:
                        single_channels.append(channel_h5subgroup[key])
                di_signals.append(single_channels)
        out = np.array(di_signals[0][0]['signal_full'])
        measurement_data_original = out.reshape(out.shape[2], out.shape[0]*out.shape[1])
    return measurement_data_original

rec_1 = load_recordings(measurement_dict_2020_140_PA_ch3)
sample_frequency = 1e4
t_ch1 = np.linspace(0, rec_1.shape[1]/sample_frequency, rec_1.shape[1]) * 1e3
rec_1_mean = rec_1.mean(axis=0)

# todo check tms peak and check dt here
# where does the peak go?, check ustils version of this...
tms_peak_idx = scipy.signal.find_peaks(rec_1_mean, height=rec_1_mean.max()*0.9)[0][0]
t_min1ms_idx = int(tms_peak_idx - (5 / dt))
t_2ms_idx = int(tms_peak_idx + (2 / dt))
t_15ms_idx = int(tms_peak_idx + (15 / dt))
t_5ms_idx = int(tms_peak_idx + (5 / dt))
rec_1_mean = scipy.ndimage.gaussian_filter1d(rec_1_mean, sigma=1)
for i in range(rec_1.shape[0]):
    plt.plot(rec_1[i, t_min1ms_idx:t_15ms_idx], color='k', alpha=0.3)
plt.plot(rec_1_mean[t_min1ms_idx:t_15ms_idx], color='blue', alpha=1.0)
a=1
