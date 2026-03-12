import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
from Model.Neck import generate_EP
import scipy
from Utils import get_peak_values, argmin_2d, butter_highpass_filter, get_I_wave_locs
import h5py
from tqdm.contrib import itertools
from Model.Nykamp_Model import Nykamp_Model_1
matplotlib.use('TkAgg')

dt = 0.01
dv = 0.01
T = 14

highpass = True
section = False
make_hist_figs = False

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
measurement_dict_2004_155_PA_1_ch2 = dict(orientation='PA', threshold=155, year=2004, hdf5_path=hdf5_path, sigma=0.1)
measurement_dict_2013_120_PA_ch2 = dict(orientation='PA', threshold=120, year=2013, hdf5_path=hdf5_path, sigma=0.1)

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

data_dicts = [measurement_dict_2020_100_PA_ch3,
              measurement_dict_2013_110_PA_ch2,
              measurement_dict_2013_110_PA_ch3,
              measurement_dict_2020_120_PA_ch3,
              measurement_dict_2007_120_PA_ch3,
              measurement_dict_2013_120_PA_ch2,
              measurement_dict_2020_140_PA_ch3,
              measurement_dict_2004_146_PA_2_ch2,
              measurement_dict_2004_154_PA_2_ch2,
              measurement_dict_2007_150_PA_ch3,
              measurement_dict_2004_150_PA_1_ch2,
              measurement_dict_2004_155_PA_1_ch2]

data_dicts_LM = [measurement_dict_2020_80_LM_ch3,
              measurement_dict_2020_80_LM_ch4,
              measurement_dict_2020_100_LM_ch3,
              measurement_dict_2020_100_LM_ch4,
              measurement_dict_2013_100_LM_ch2,
              measurement_dict_2013_100_LM_ch3,
              measurement_dict_2020_120_LM_ch3,
              measurement_dict_2020_120_LM_ch4,
              measurement_dict_2007_120_LM_ch2,
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
        measurement_data_original = out[0].T
    return measurement_data_original




# recording = load_recordings(measurement_dict_2007_150_PA_ch3)
all_dicts = data_dicts + data_dicts_LM
if make_hist_figs:
    for k, dict_k in enumerate(all_dicts):

        recording = load_recordings(dict_k)
        sample_frequency = 1e4
        t, dt, tms_idxs, time_intervals, I_wave_times, I_wave_amps, rec_local = get_I_wave_locs(recording,
                                                                                                sample_frequency=sample_frequency)
        tms_peak_idx, t_min1ms_idx, t_2ms_idx, t_5ms_idx = tms_idxs
        tI1, tI2, tI3, tI4, tI5, tI6 = time_intervals
        I1_times, I2_times, I3_times = I_wave_times
        I1_amps, I2_amps, I3_amps = I_wave_amps
        rec_local, rec_mean_local = rec_local
        fig = plt.figure(figsize=(10, 3))
        fig.suptitle(
            f'{dict_k["orientation"]} - {dict_k["threshold"]} - {recording.shape[0]} samples - {dict_k["year"]}')
        ax = fig.add_subplot(131)

        for i in range(recording.shape[0]):
            plt.plot(t, rec_local[i], color='k', alpha=0.2)
        ax.plot(t, rec_mean_local, color='blue', alpha=1.0)
        ax.set_ylim(-1.5, 5)

        shade_bot = np.ones_like(t)*-30
        shade_top = np.ones_like(t)*-30
        shade_top[tI1:tI2] = 30
        shade_top[tI3:tI4] = 30
        shade_top[tI5:tI6] = 30
        ax.fill_between(t, shade_top, shade_bot, alpha=0.3, color='k', zorder=-10)
        ax.set_xlim(-1, 10)
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('v (µV)')
        ax.text(2.5, 4.0, 'D')
        ax.text(4.0, 4.0, 'I1')
        ax.text(5.5, 4.0, 'I2')

        ax = fig.add_subplot(132)
        # plot distribution of peaks in I1, I2, I3 and D interval

        counts, bins = np.histogram(I1_amps)
        ax.stairs(counts, bins, fill=True, color='indianred', alpha=0.3)
        counts, bins = np.histogram(I2_amps)
        ax.stairs(counts, bins, fill=True, color='darkorange', alpha=0.3)
        counts, bins = np.histogram(I3_amps)
        ax.stairs(counts, bins, fill=True, color='teal', alpha=0.3)
        ax.legend(['D-wave peaks', 'I1-peaks', 'I2-peaks'])

        ax.set_xlabel('Amplitude (µV)')
        ax.set_ylabel('Histogram')

        ax = fig.add_subplot(133)
        # plot distribution of peaks in I1, I2, I3 and D interval

        counts, bins = np.histogram(I1_times, bins=np.arange(I1_times.min(), I1_times.max()+dt, dt))
        ax.stairs(counts, bins, fill=True, color='indianred', alpha=0.3)
        counts, bins = np.histogram(I2_times, bins=np.arange(I2_times.min(), I2_times.max()+dt, dt))
        ax.stairs(counts, bins, fill=True, color='darkorange', alpha=0.3)
        counts, bins = np.histogram(I3_times, bins=np.arange(I3_times.min(), I3_times.max()+dt, dt))
        ax.stairs(counts, bins, fill=True, color='teal', alpha=0.3)
        ax.legend(['D-wave peaks', 'I1-peaks', 'I2-peaks'], loc='upper right')
        ax.set_xlabel('Amplitude time (ms)')
        ax.set_ylabel('Histogram')
        plt.tight_layout()

        # plt.show()
        plt.savefig(f'I-wave_hists_{k}.png', dpi=200)
        print(f'saved to I-wave_hists_{k}.png')
        a=1

make_boxplots = True
pa_box_dicts = [measurement_dict_2020_100_PA_ch3,
                measurement_dict_2013_110_PA_ch2,
                measurement_dict_2013_120_PA_ch2,
                measurement_dict_2020_140_PA_ch3,
                measurement_dict_2007_150_PA_ch3]
if make_boxplots:
    for k, dict_k in enumerate(pa_box_dicts):
        sample_frequency=1e4
        t, dt, tms_idxs, time_intervals, I_wave_times, I_wave_amps, rec_local = get_I_wave_locs(recording,
                                                                                                sample_frequency=sample_frequency)
