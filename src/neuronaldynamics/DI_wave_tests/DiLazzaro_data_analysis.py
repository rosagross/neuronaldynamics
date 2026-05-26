import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from Utils import get_I_wave_locs, argmin_2d, butter_highpass_filter, get_peak_values
from Model import generate_EP
import h5py
from tqdm.contrib import itertools

matplotlib.use('TkAgg')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

dt = 0.01
dv = 0.01
T = 14

highpass = True
section = False
make_hist_figs = False
make_boxplots = True

t = np.arange(0, T, dt)
Nt = t.shape[0]
detrend = False

# fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
# hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
if not detrend:
    # hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
    hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
else:
    # hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data_detrended.hdf5"
    hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data_detrended.hdf5"


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
        sample_frequency = np.array(di_signals[0][0]['sample_frequency'])

    return measurement_data_original, sample_frequency




all_dicts = data_dicts + data_dicts_LM
# all_dicts = [measurement_dict_2004_140_LM_1_ch2]
if make_hist_figs:
    for k, dict_k in enumerate(all_dicts):
        dtI = 1.5
        tD = 2.0
        if dict_k['year'] == 2004 and dict_k['orientation'] == 'LM':
            dtI = 1.2
            tD = 3.0
        recording, sample_frequency = load_recordings(dict_k)
        t, dt, tms_idxs, time_intervals, I_wave_times, I_wave_amps, rec_local = get_I_wave_locs(recording,
                                                                                                sample_frequency=sample_frequency,
                                                                                                dtI=dtI,
                                                                                                tD=tD)
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
        ax.text(tD + 0.5, 4.0, 'D')
        ax.text(tD + dtI + 0.5, 4.0, 'I1')
        ax.text(tD + 2*dtI + 0.5, 4.0, 'I2')

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

        counts, bins = np.histogram(I1_times, bins=np.arange(I1_times.min()-dt, I1_times.max()+dt, dt))
        ax.stairs(counts, bins, fill=True, color='indianred', alpha=0.3)
        counts, bins = np.histogram(I2_times, bins=np.arange(I2_times.min()-dt, I2_times.max()+dt, dt))
        ax.stairs(counts, bins, fill=True, color='darkorange', alpha=0.3)
        counts, bins = np.histogram(I3_times, bins=np.arange(I3_times.min()-dt, I3_times.max()+dt, dt))
        ax.stairs(counts, bins, fill=True, color='teal', alpha=0.3)
        ax.legend(['D-wave peaks', 'I1-peaks', 'I2-peaks'], loc='upper right')
        ax.set_xlabel('Amplitude time (ms)')
        ax.set_ylabel('Histogram')
        plt.tight_layout()

        # plt.show()
        plt.savefig(f'I-wave_hists_{k}.png', dpi=200)
        print(f'saved to I-wave_hists_{k}.png')
        plt.close()

pa_box_dicts = [measurement_dict_2020_100_PA_ch3,
                measurement_dict_2013_110_PA_ch2,
                measurement_dict_2013_120_PA_ch2,
                measurement_dict_2020_140_PA_ch3]
pa_thresholds = [100, 110, 120, 140]
lm_box_dicts = [measurement_dict_2020_120_LM_ch3,
                measurement_dict_2004_140_LM_1_ch2]
lm_thresholds = [120, 140]

if make_boxplots:
    # TODO: rename the I1 data to D wave, since it's the D wave
    I1_wave_times_pa = []
    I2_wave_times_pa = []
    I3_wave_times_pa = []

    I1_wave_times_lm = []
    I2_wave_times_lm = []
    I3_wave_times_lm = []

    for k, dict_k in enumerate(pa_box_dicts):
        recording, sample_frequency = load_recordings(dict_k)
        t, dt, tms_idxs, time_intervals, I_wave_times, I_wave_amps, rec_local = get_I_wave_locs(recording,
                                                                                                sample_frequency=sample_frequency)
        I1_wave_times_pa.append(I_wave_times[0])
        I2_wave_times_pa.append(I_wave_times[1])
        I3_wave_times_pa.append(I_wave_times[2])

    # lm 120
    recording, sample_frequency = load_recordings(lm_box_dicts[0])
    t, dt, tms_idxs, time_intervals, I_wave_times, I_wave_amps, rec_local = get_I_wave_locs(recording,
                                                                                            sample_frequency=sample_frequency)
    I1_wave_times_lm.append(I_wave_times[0])
    I2_wave_times_lm.append(I_wave_times[1])
    I3_wave_times_lm.append(I_wave_times[2])

    # lm 140
    recording, sample_frequency = load_recordings(lm_box_dicts[1])
    t, dt, tms_idxs, time_intervals, I_wave_times, I_wave_amps, rec_local = get_I_wave_locs(recording,
                                                                                            sample_frequency=sample_frequency,
                                                                                            dtI=1.2,
                                                                                            tD=3.0)
    I1_wave_times_lm.append(I_wave_times[0])
    I2_wave_times_lm.append(I_wave_times[1])
    I3_wave_times_lm.append(I_wave_times[2])

    # calculate means
    I1_pa_time_means = np.zeros(len(I1_wave_times_pa))
    I2_pa_time_means = np.zeros_like(I1_pa_time_means)
    I3_pa_time_means = np.zeros_like(I1_pa_time_means)
    for j in range(len(I1_wave_times_pa)):
        I1_pa_time_means[j] = I1_wave_times_pa[j].mean()
        I2_pa_time_means[j] = I2_wave_times_pa[j].mean()
        I3_pa_time_means[j] = I3_wave_times_pa[j].mean()

    I1_lm_time_means = np.zeros(len(I1_wave_times_lm))
    I2_lm_time_means = np.zeros_like(I1_lm_time_means)
    I3_lm_time_means = np.zeros_like(I1_lm_time_means)
    for j in range(len(I1_wave_times_lm)):
        I1_lm_time_means[j] = I1_wave_times_lm[j].mean()
        I2_lm_time_means[j] = I2_wave_times_lm[j].mean()
        I3_lm_time_means[j] = I3_wave_times_lm[j].mean()

    ####################################################################################################################
    # Look for optimal map of E-field to RMT value
    ####################################################################################################################
    with h5py.File('E_theta_2D_frexc_pt6.hdf5', 'r') as h5file:
        data = np.array(h5file['E_theta_2D']) * 1e3  # conversion from 1/ms to 1/s

    E_values = np.linspace(150, 400, 100)
    theta_values = np.linspace(0, 180, 100)
    E_mesh, theta_mesh = np.meshgrid(E_values, theta_values)
    mesh_shapes = E_mesh.shape
    dt = 0.01
    dv = 0.01
    T = 14
    t = np.arange(0, T, dt)
    height = 0.5

    for i, theta_i in enumerate(theta_values):
        EP, t_EP, AP_out = generate_EP(d=0.1, plot=False, Axontype=1, dt=dt * 10)
        EP = -EP
        EP = EP / np.max(EP)
        EP_small = np.interp(t[t < 1.0] - 0.5, t_EP, EP)
        for j in range(data[i].shape[0]):
            nmm_potential = scipy.signal.convolve(data[i, j], EP_small)
            nmm_shape = data[i, j].shape[0]
            nmm_potential_out = nmm_potential[:nmm_shape]

            v_out_hp = butter_highpass_filter(nmm_potential_out, cutoff=0.05,
                                              fps=int(1 / dt))  # very small cutoff
            v_out_mean = nmm_potential_out.mean()
            data[i, j] = v_out_hp

    theta_start = 0
    theta_end = 120
    theta_idx_start = np.where(theta_values > theta_start)[0][0] - 1
    theta_idx_end = np.where(theta_values > theta_end)[0][0]
    theta_range = theta_values[theta_idx_start:theta_idx_end]

    dt_I12 = np.zeros((theta_range.shape[0], E_values.shape[0]))
    dt_I23 = np.zeros_like(dt_I12)
    dt_I34 = np.zeros_like(dt_I12)
    tI1 = np.zeros_like(dt_I12)

    for i in range(theta_range.shape[0]):
        for j in range(E_values.shape[0]):
            peak_values_j = get_peak_values(t, data[i, j], find_peak_args=dict(height=height))
            n_iwaves_j = peak_values_j['t_delta_peaks'].shape[0]
            if n_iwaves_j < 1:
                tI1[i, j] = np.nan
            else:
                tI1[i, j] = peak_values_j['peak_1_time']
            if n_iwaves_j < 1:
                dt_I12[i, j] = np.nan
            else:
                dt_I12[i, j] = peak_values_j['t_delta_peaks'][0]
            if n_iwaves_j < 2:
                dt_I23[i, j] = np.nan
            else:
                dt_I23[i, j] = peak_values_j['t_delta_peaks'][1]
            if n_iwaves_j < 3:
                dt_I34[i, j] = np.nan
            else:
                dt_I34[i, j] = peak_values_j['t_delta_peaks'][1]

    delay = 0.9  # 1.3
    rmt_array = np.array(pa_thresholds)
    a_values = np.linspace(1.0, 2.3, 100)
    b_values = np.linspace(0, 50, 100)
    sqerror_theta = np.zeros((theta_range.shape[0], 100, 100))
    a_opt = np.zeros(theta_range.shape[0])
    b_opt = np.zeros(theta_range.shape[0])
    opt_idxs = np.zeros((theta_range.shape[0], 2), dtype=np.int64)
    min_sqerror = np.zeros(theta_range.shape[0])
    sqerror = np.zeros((theta_range.shape[0], 100, 100))
    tI1_model = tI1 + delay
    tI2_model = tI1 + delay + dt_I12
    tI3_model = tI1 + delay + dt_I12 + dt_I23

    print('performing extensive grid search for PA fits')
    for m, i, j in itertools.product(range(theta_range.shape[0]), range(a_values.shape[0]), range(b_values.shape[0])):

        a = a_values[i]
        b = b_values[j]

        E_map = a * rmt_array + b
        E_idxs_data = np.zeros(E_map.shape[0], dtype=np.int64)

        for k in range(len(I1_wave_times_pa)):
            E_idxs_data[k] = np.floor(np.where(E_values > E_map[k])[0][0])
        # important I1 data is actually D wave
        dy_t1 = np.sum((tI1_model[m, E_idxs_data] - I2_pa_time_means) ** 2)
        dy_t2 = np.sum((tI2_model[m, E_idxs_data] - I3_pa_time_means) ** 2)
        # dy_t3 = np.sum((tI3_model[m, E_idxs_data] - I3_pa_time_means) ** 2)

        sum_dy = dy_t1 + dy_t2

        sqerror[m, i, j] = sum_dy

        if i == a_values.shape[0] - 1 and j == b_values.shape[0] - 1:
            opt_idxs[m] = argmin_2d(sqerror[m])
            min_sqerror[m] = np.nanmin(sqerror[m])
            a_opt[m] = a_values[opt_idxs[m, 0]]
            b_opt[m] = b_values[opt_idxs[m, 1]]
        # E_map_opt = a_opt[k]*rmt_array + b_opt[k]
    opt_theta_idx = np.argmin(min_sqerror)
    a_theta_opt = a_opt[opt_theta_idx]
    b_theta_opt = b_opt[opt_theta_idx]
    E_map_opt_pa = a_theta_opt * rmt_array + b_theta_opt
    pa_theta = theta_range[opt_theta_idx]
    pa_tI1 = tI1_model[opt_theta_idx]
    pa_tI2 = tI2_model[opt_theta_idx]
    print(f'opt angle pa: {pa_theta}')

    # lm
    theta_start = 30
    theta_end = 150
    theta_idx_start = np.where(theta_values > theta_start)[0][0] - 1
    theta_idx_end = np.where(theta_values > theta_end)[0][0]
    theta_range = theta_values[theta_idx_start:theta_idx_end]

    dt_I12 = np.zeros((theta_range.shape[0], E_values.shape[0]))
    dt_I23 = np.zeros_like(dt_I12)
    dt_I34 = np.zeros_like(dt_I12)
    tI1 = np.zeros_like(dt_I12)

    for i in range(theta_range.shape[0]):
        for j in range(E_values.shape[0]):
            peak_values_j = get_peak_values(t, data[i, j], find_peak_args=dict(height=height))
            n_iwaves_j = peak_values_j['t_delta_peaks'].shape[0]
            if n_iwaves_j < 1:
                tI1[i, j] = np.nan
            else:
                tI1[i, j] = peak_values_j['peak_1_time']
            if n_iwaves_j < 1:
                dt_I12[i, j] = np.nan
            else:
                dt_I12[i, j] = peak_values_j['t_delta_peaks'][0]
            if n_iwaves_j < 2:
                dt_I23[i, j] = np.nan
            else:
                dt_I23[i, j] = peak_values_j['t_delta_peaks'][1]
            if n_iwaves_j < 3:
                dt_I34[i, j] = np.nan
            else:
                dt_I34[i, j] = peak_values_j['t_delta_peaks'][1]
    rmt_array = np.array(lm_thresholds)
    a_values = np.linspace(1.0, 2.3, 100)
    b_values = np.linspace(0, 50, 100)
    sqerror_theta = np.zeros((theta_range.shape[0], 100, 100))
    a_opt = np.zeros(theta_range.shape[0])
    b_opt = np.zeros(theta_range.shape[0])
    opt_idxs = np.zeros((theta_range.shape[0], 2), dtype=np.int64)
    min_sqerror = np.zeros(theta_range.shape[0])
    sqerror = np.zeros((theta_range.shape[0], 100, 100))
    tI1_model = tI1 + delay
    tI2_model = tI1 + delay + dt_I12
    tI3_model = tI1 + delay + dt_I12 + dt_I23

    print('performing extensive grid search for LM fits')
    for m, i, j in itertools.product(range(theta_range.shape[0]), range(a_values.shape[0]), range(b_values.shape[0])):

        a = a_values[i]
        b = b_values[j]

        E_map = a * rmt_array + b
        E_idxs_data = np.zeros(E_map.shape[0], dtype=np.int64)

        for k in range(len(I1_wave_times_lm)):
            E_idxs_data[k] = np.floor(np.where(E_values > E_map[k])[0][0])
        # important I1 data is actually D wave
        dy_t1 = np.sum((tI1_model[m, E_idxs_data] - I2_lm_time_means) ** 2)
        dy_t2 = np.sum((tI2_model[m, E_idxs_data] - I3_lm_time_means) ** 2)
        # dy_t3 = np.sum((tI3_model[m, E_idxs_data] - I3_pa_time_means) ** 2)

        sum_dy = dy_t1 + dy_t2

        sqerror[m, i, j] = sum_dy

        if i == a_values.shape[0] - 1 and j == b_values.shape[0] - 1:
            opt_idxs[m] = argmin_2d(sqerror[m])
            min_sqerror[m] = np.nanmin(sqerror[m])
            a_opt[m] = a_values[opt_idxs[m, 0]]
            b_opt[m] = b_values[opt_idxs[m, 1]]
        # E_map_opt = a_opt[k]*rmt_array + b_opt[k]
    opt_theta_idx = np.argmin(min_sqerror)
    a_theta_opt = a_opt[opt_theta_idx]
    b_theta_opt = b_opt[opt_theta_idx]
    E_map_opt_lm = a_theta_opt * rmt_array + b_theta_opt

    lm_theta = theta_range[opt_theta_idx]
    lm_tI1 = tI1_model[opt_theta_idx]
    lm_tI2 = tI2_model[opt_theta_idx]
    print(f'opt angle lm: {lm_theta}')

    ####################################################################################################################
    # Do Boxplots
    ####################################################################################################################
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(231)
    ax.boxplot(I1_wave_times_pa)
    ax.set_xlabel('% RMT')
    ax.set_xticklabels(pa_thresholds)
    ax.set_title('D-wave time')
    ax.set_ylabel('t (ms)')

    ax = fig.add_subplot(232)
    ax.boxplot(I2_wave_times_pa)
    ax.set_xlabel('% RMT')
    ax.set_xticklabels(pa_thresholds)
    ax.set_title('I1-wave time')
    ax.set_ylabel('t (ms)')

    ax = fig.add_subplot(233)
    ax.boxplot(I3_wave_times_pa)
    ax.set_xlabel('% RMT')
    ax.set_xticklabels(pa_thresholds)
    ax.set_title('I2-wave time')
    ax.set_ylabel('t (ms)')

    ax = fig.add_subplot(234)
    ax.boxplot(I1_wave_times_lm)
    ax.set_xlabel('% RMT')
    ax.set_xticklabels(lm_thresholds)
    ax.set_title('D-wave time')
    ax.set_ylabel('t (ms)')

    ax = fig.add_subplot(235)
    ax.boxplot(I2_wave_times_lm)
    ax.set_xlabel('% RMT')
    ax.set_xticklabels(lm_thresholds)
    ax.set_title('I1-wave time')
    ax.set_ylabel('t (ms)')

    ax = fig.add_subplot(236)
    ax.boxplot(I3_wave_times_lm)
    ax.set_xlabel('% RMT')
    ax.set_xticklabels(lm_thresholds)
    ax.set_title('I2-wave time')
    ax.set_ylabel('t (ms)')

    plt.tight_layout()
    plt.savefig('Iwave_latency_boxplots.png')
    print(f'saved img to Iwave_latency_boxplots.png')
    # plt.show()

    # box plots with E field model data
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(231)
    ax.boxplot(I1_wave_times_pa, positions=E_map_opt_pa, widths=20)
    ax.set_xticklabels([f'{k:.0f}' for k in E_map_opt_pa])
    ax.set_xlabel('|E| V/m')
    # ax.set_xticklabels(pa_thresholds)
    # ax.set_xlabel('% RMT')
    ax.set_xlim(200, 400)
    ax.set_title('D-wave time')
    ax.set_ylabel('t (ms)')
    ax.text(150, 2.7, 'PA', rotation=90, fontsize=14)

    ax = fig.add_subplot(232)
    ax.boxplot(I2_wave_times_pa, positions=E_map_opt_pa, widths=20)
    ax.set_xticklabels([f'{k:.0f}' for k in E_map_opt_pa])
    ax.set_xlabel('|E| V/m')
    # ax.set_xticklabels(pa_thresholds)
    # ax.set_xlabel('% RMT')
    ax.set_xlim(200, 400)
    ax.plot(E_values, pa_tI1, label='model')
    ax.legend()
    ax.set_ylim(3, 6.5)
    ax.set_title('I1-wave time')
    ax.set_ylabel('t (ms)')

    ax = fig.add_subplot(233)
    ax.boxplot(I3_wave_times_pa, positions=E_map_opt_pa, widths=20)
    ax.set_xticklabels([f'{k:.0f}' for k in E_map_opt_pa])
    ax.set_xlabel('|E| V/m')
    # ax.set_xticklabels(pa_thresholds)
    # ax.set_xlabel('% RMT')
    ax.set_xlim(200, 400)
    ax.plot(E_values, pa_tI2, label='model')
    ax.legend()
    ax.set_ylim(5, 7)
    ax.set_title('I2-wave time')
    ax.set_ylabel('t (ms)')


    # lm

    ax = fig.add_subplot(234)
    ax.boxplot(I1_wave_times_lm, positions=E_map_opt_lm, widths=20)
    ax.set_xticklabels([f'{k:.0f}, {lm_thresholds[i]}' for i, k in enumerate(E_map_opt_lm)])
    ax.set_xlabel('|E| V/m')
    # ax.set_xticklabels(lm_thresholds)
    # ax.set_xlabel('% RMT')
    ax.set_xlim(200, 400)
    ax.set_title('D-wave time')
    ax.set_ylabel('t (ms)')

    ax.text(150, 2.95, 'LM', rotation=90, fontsize=14)

    ax = fig.add_subplot(235)
    ax.boxplot(I2_wave_times_lm, positions=E_map_opt_lm, widths=20)
    ax.set_xticklabels([f'{k:.0f}' for k in E_map_opt_lm])
    ax.set_xlabel('|E| V/m')
    # ax.set_xticklabels(lm_thresholds)
    # ax.set_xlabel('% RMT')
    ax.set_xlim(200, 400)
    ax.plot(E_values, lm_tI1, label='model')
    ax.legend()
    ax.set_ylim(3, 6.5)
    ax.set_title('I1-wave time')
    ax.set_ylabel('t (ms)')

    ax = fig.add_subplot(236)
    ax.boxplot(I3_wave_times_lm, positions=E_map_opt_lm, widths=20)
    ax.set_xticklabels([f'{k:.0f}' for k in E_map_opt_lm])
    ax.set_xlabel('|E| V/m')
    # ax.set_xticklabels(lm_thresholds)
    # ax.set_xlabel('% RMT')
    ax.set_xlim(200, 400)
    ax.set_ylim(5, 7)
    ax.plot(E_values, lm_tI2, label='model')
    ax.legend()
    ax.set_title('I2-wave time')
    ax.set_ylabel('t (ms)')



    plt.tight_layout()
    plt.subplots_adjust(left=0.10)

    plt.savefig('Iwave_e_field_boxplots.png')
    print(f'saved img to Iwave_e_field_boxplots.png')