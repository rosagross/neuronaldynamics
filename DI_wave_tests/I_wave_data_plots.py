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
plot_overview = False

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
measurement_dict = dict(orientation='PA', threshold=120, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=0)
if detrend:
    hp = False
else:
    hp = True
parameters = {'intensity': 250, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.5, 'plot_align': False,
              'test_func_intensity': 2.0, 'test_func_t0': 0.35, 'enable_high_pass': False, 'min_delay': 5,
              'test_signal_from_file': True, 'i_scale': 5.148136e-6, 'detrend': detrend, 'plot_detrend': False,
              'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'mind_delay': 0,
              'theta': 90,
              'file_args': measurement_dict_2020_140_PA_ch3,
              'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
                                    'tau_ref': [0], #1.5
                                    'tau_mem': [12],
                                    'input_type': 'stochastic-current',
                                    'static_noise': True,
                                    'init_pdf_offset': 0,
                                    'init_pdf_sigma': 0.5,
                                    'init_pdf_weight': 0,
                                    'delay_kernel_type': 'alpha',
                                    'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3},
                                    'dv': dv,
                                    'dt': dt,
                                    'solver': 'Hu-2021',
                                    'current_sigma': 4,
                                    'verbose': 1}}
pa_100 = []
pa_110 = []
pa_120 = []
pa_140 = []
pa_150 = []
max_pa_amp=0
for i, dict_i in enumerate(data_dicts):
    parameters['file_args'] = dict_i
    di_model = DI_wave_simulation(parameters=parameters, logname=None)
    di_model.get_test_signal(plot_d_wave_detection=False, from_file=True, hdf5_args=di_model.file_args, highpass=hp)
    iwaves = di_model.target
    peak_min_dist = int(1/dt)
    if iwaves.max() > max_pa_amp:
        max_pa_amp = iwaves.max()
    if dict_i['threshold'] < 110:
        pa_100.append(iwaves)
    elif dict_i['threshold'] >100 and dict_i['threshold'] <120:
        pa_110.append(iwaves)
    elif dict_i['threshold'] > 110 and dict_i['threshold'] < 130:
        pa_120.append(iwaves)
    elif dict_i['threshold'] >130 and dict_i['threshold'] <145:
        pa_140.append(iwaves)
    elif dict_i['threshold'] >144 and dict_i['threshold'] <160:
        pa_150.append(iwaves)

    find_peak_args = {'distance':peak_min_dist, 'height': 0.5}
    # peak_vals = get_peak_values(x=t, y=iwaves, plot=True, find_peak_args=find_peak_args)

if not scale:
    max_pa_amp = 1
pa_lists = [pa_100, pa_110, pa_120, pa_140, pa_150]
pa_names = ['PA- 100%RMT', 'PA- 110%RMT', 'PA- 120%RMT', 'PA- 140%RMT', 'PA- 150%RMT']
if plot_overview:
    fig = plt.figure(figsize=(6, 10))
    for j in range(len(pa_lists)):
        ax = fig.add_subplot(len(pa_lists), 1, j+1)
        for l in range(len(pa_lists[j])):
            ax.plot(t, pa_lists[j][l]/max_pa_amp, label=pa_names[l])
            ax.set_xlim((0, 12))
            # ax.text(f'number of samples: {len(pa_lists[j])}')
        if j == len(pa_lists) - 1:
            ax.set_xlabel('t (ms)')
        ax.set_ylabel(pa_names[j])
    plt.tight_layout()
    plt.show()

lm_80 = []
lm_100 = []
lm_120 = []
lm_140 = []
max_lm_amp = 0
for i, dict_i in enumerate(data_dicts_LM):
    parameters['file_args'] = dict_i
    di_model = DI_wave_simulation(parameters=parameters, logname=None)
    di_model.get_test_signal(plot_d_wave_detection=False, from_file=True, hdf5_args=di_model.file_args, highpass=hp)
    iwaves = di_model.target
    peak_min_dist = int(1/dt)
    if iwaves.max() > max_lm_amp:
        max_lm_amp = iwaves.max()
    if dict_i['threshold'] < 90:
        lm_80.append(iwaves)

    elif dict_i['threshold'] >90 and dict_i['threshold'] <115:
        lm_100.append(iwaves)
    elif dict_i['threshold'] > 115 and dict_i['threshold'] < 125:
        lm_120.append(iwaves)
    elif dict_i['threshold'] >135 and dict_i['threshold'] <145:
        lm_140.append(iwaves)

    find_peak_args = {'distance':peak_min_dist, 'height': 0.5}
    # peak_vals = get_peak_values(x=t, y=iwaves, plot=True, find_peak_args=find_peak_args)
if not scale:
    max_lm_amp = 1
lm_lists = [lm_80, lm_100, lm_120, lm_140]
lm_names = ['LM- 80%RMT', 'LM- 100%RMT', 'LM- 120%RMT', 'LM- 140%RMT']
if plot_overview:
    fig = plt.figure(figsize=(6, 10))
    for j in range(len(lm_lists)):
        ax = fig.add_subplot(len(lm_lists), 1, j+1)
        for l in range(len(lm_lists[j])):
            ax.plot(t, lm_lists[j][l]/max_lm_amp, label=lm_names[l])
            ax.set_xlim((0, 12))
            # ax.text(f'number of samples: {len(pa_lists[j])}')
        if j == len(lm_lists) - 1:
            ax.set_xlabel('t (ms)')
        ax.set_ylabel(lm_names[j])
    plt.tight_layout()
    plt.show()


########################################################################################################################
# PA Plots
########################################################################################################################


with h5py.File('E_theta_2D.hdf5', 'r') as h5file:
    data = np.array(h5file['E_theta_2D']) * 1e3  # conversion from 1/ms to 1/s



E_values = np.linspace(150, 400, 100)
theta_values = np.linspace(0, 180, 100)
E_mesh, theta_mesh = np.meshgrid(E_values, theta_values)
mesh_shapes = E_mesh.shape

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
        data[i, j] = v_out_hp / 1000

theta_guess = 20
theta_idx = np.where(theta_values > theta_guess)[0][0]
n_pa = len(pa_lists)
dt_I12 = np.zeros((E_values.shape[0]))
dt_I23 = np.zeros_like(dt_I12)
dt_I34 = np.zeros_like(dt_I12)
dA_I12 = np.zeros_like(dt_I12)
dA_I23 = np.zeros_like(dt_I12)
dA_I34 = np.zeros_like(dt_I12)
tI1 = np.zeros_like(dt_I12)
amp_max = np.zeros_like(dt_I12)
I_area = np.zeros_like(dt_I12)
n_iwaves = np.zeros_like(dt_I12)


# E_idx = np.where(E_values > E_plot[1])[0][0]

i = theta_idx
for j in range(E_values.shape[0]):
    peak_values_j = get_peak_values(t, data[i, j], find_peak_args=dict(height=height))
    n_iwaves_j = peak_values_j['t_delta_peaks'].shape[0]
    n_iwaves[j] = n_iwaves_j
    if n_iwaves_j < 1:
        tI1[j] = np.nan
        amp_max[j] = np.nan
        I_area[j] = np.nan
    else:
        tI1[j] = peak_values_j['peak_1_time']
        amp_max[j] = peak_values_j['peak_max_amp']
        I_area[j] = peak_values_j['area']
    if n_iwaves_j < 1:
        dt_I12[j] = np.nan
        dA_I12[j] = np.nan
    else:
        dt_I12[j] = peak_values_j['t_delta_peaks'][0]
        dA_I12[j] = peak_values_j['amp_delta_peaks'][0]
    if n_iwaves_j < 2:
        dt_I23[j] = np.nan
        dA_I23[j] = np.nan
    else:
        dt_I23[j] = peak_values_j['t_delta_peaks'][1]
        dA_I23[j] = peak_values_j['amp_delta_peaks'][1]
    if n_iwaves_j < 3:
        dt_I34[j] = np.nan
        dA_I34[j] = np.nan
    else:
        dt_I34[j] = peak_values_j['t_delta_peaks'][1]
        dA_I34[j] = peak_values_j['amp_delta_peaks'][1]

# generate signal from data
# convoluted nested lists atm, think of something better eventually?
pa_thresholds = [100, 110, 120, 140, 150]
mean_pa_signals = np.zeros((5, t.shape[0]))
pa_signals = []
dtI12_data = np.zeros(5)
dtI23_data = np.zeros(5)
dtI34_data = np.zeros(5)
tI1_data = np.zeros(5)
n_iwaves_data = np.zeros(5)
for i in range(n_pa):
    pa_list_threshold=[]
    pa_signals.append([])
    for j in range(len(pa_lists[i])):
        if pa_lists[i][j].max() > 2:
            pa_list_threshold.append(pa_lists[i][j])

    pa_signals.append(pa_list_threshold)
    mean_pa_signals[i] = np.mean(np.array(pa_list_threshold), axis=0)

    peak_values_ij = get_peak_values(t,mean_pa_signals[i], find_peak_args=dict(height=1))
    n_iwaves_i = peak_values_ij['t_delta_peaks'].shape[0]
    n_iwaves_data[i] = n_iwaves_i
    if n_iwaves_i < 1:
        tI1_data[i] = np.nan
        # amp_max[i, j] = np.nan
        # I_area[i, j] = np.nan
    else:
        tI1_data[i] = peak_values_ij['peak_1_time']
        dtI12_data[i] = peak_values_ij['t_delta_peaks'][0]
    if n_iwaves_i < 2:
        dtI23_data[i] = np.nan
    else:
        dtI23_data[i] = peak_values_ij['t_delta_peaks'][1]
    if n_iwaves_i < 3:
        dtI34_data[i] = np.nan
    else:
        dtI34_data[i] = peak_values_ij['t_delta_peaks'][2]



# map from data pts to E-field
delay = 0.7 #1.3
tI_delayed = tI1 + delay
rmt_array = np.array(pa_thresholds)
a_values = np.linspace(1.0, 2.3, 100)
b_values = np.linspace(0, 50, 100)
sqerror = np.zeros((100, 100))
for i, j in itertools.product(range(a_values.shape[0]), range(b_values.shape[0])):
    # a = 2
    # b= 50
    a = a_values[i]
    b = b_values[j]

    E_map = a*rmt_array + b
    E_idxs_data = np.zeros(E_map.shape[0], dtype=np.int64)
    for k in range(n_pa):
        E_idxs_data[k] = np.floor(np.where(E_values > E_map[k])[0][0])

    dy = (tI_delayed[E_idxs_data] - tI1_data)**2
    sqerror[i, j] = np.sum(dy)
min_sqerror_idxs = argmin_2d(sqerror)
min_sqerror = sqerror.min()
opt_ab = np.zeros((1, 2))
a_opt = a_values[min_sqerror_idxs[0]]
b_opt = b_values[min_sqerror_idxs[1]]
opt_ab[0] = np.array((a_opt, b_opt))
E_map_opt = a_opt*rmt_array + b_opt
fig = plt.figure(figsize=(8, 2))

ax = fig.add_subplot(1, 4, 1)
ax.plot(E_values, tI_delayed)
ax.scatter(E_map_opt, tI1_data, marker='x', color='k')

ax = fig.add_subplot(1, 4, 2)
ax.plot(E_values, dt_I12)
ax.scatter(E_map_opt, dtI12_data, marker='x', color='k')

ax = fig.add_subplot(1, 4, 3)
ax.plot(E_values, dt_I23)
ax.scatter(E_map_opt, dtI23_data, marker='x', color='k')

ax = fig.add_subplot(1, 4, 4)
ax.plot(E_values, dt_I34)
ax.scatter(E_map_opt, dtI34_data, marker='x', color='k')

plt.tight_layout()
plt.show()

#
# for i in range(n_pa):
#     plt.plot(t, mean_pa_signals[i])
# plt.xlim(0, 12)
# plt.xlabel('t (ms)')
# plt.ylabel('v (uV)')
# plt.grid()
# plt.legend(pa_names)
# plt.show()
#
# plt.plot(pa_thresholds, tI1_data)
# plt.plot(pa_thresholds, dtI12_data)
# plt.legend(['tI1', 'dtI12'])
# plt.show()