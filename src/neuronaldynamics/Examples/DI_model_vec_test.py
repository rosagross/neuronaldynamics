import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
from Utils import nrmse
matplotlib.use('TkAgg')


# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]

save_figs = False

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]

# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
# hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"

# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data_detrended.hdf5"
# hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"

fn_session = '/home/erik/Downloads/gpc.pkl'
hdf5_path = "/home/erik/Nextcloud_Uni/TMS Neuro Projects/M1_modeling/DI_wave_data/extracted_DI_waves/DiLazarro_di_wave_data.hdf5"

# fn_session = '/data/pt_01756/studies/DI_wave_modeling/TMS-coupling-model_gpc/gpc.pkl'
# hdf5_path = '/data/pt_01756/studies/DI_wave_modeling/DI_wave_data/DiLazarro_di_wave_data.hdf5'

simulation_name = 'vector_test_diw_sim'

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

measurement_dict_2020_80_LM_ch3 = dict(orientation='LM', threshold=80, year=2020, hdf5_path=hdf5_path, sigma=1.0,
                                       channel=0)
measurement_dict_2020_80_LM_ch4 = dict(orientation='LM', threshold=80, year=2020, hdf5_path=hdf5_path, sigma=1.0,
                                       channel=1)
measurement_dict_2020_100_LM_ch3 = dict(orientation='LM', threshold=100, year=2020, hdf5_path=hdf5_path, sigma=1.0,
                                        channel=0)
measurement_dict_2020_100_LM_ch4 = dict(orientation='LM', threshold=100, year=2020, hdf5_path=hdf5_path, sigma=1.0,
                                        channel=1)
measurement_dict_2020_120_LM_ch3 = dict(orientation='LM', threshold=120, year=2020, hdf5_path=hdf5_path, sigma=1.0,
                                        channel=0)
measurement_dict_2020_120_LM_ch4 = dict(orientation='LM', threshold=120, year=2020, hdf5_path=hdf5_path, sigma=1.0,
                                        channel=1)
measurement_dict_2013_100_LM_ch2 = dict(orientation='LM', threshold=100, year=2013, hdf5_path=hdf5_path, sigma=1.0,
                                        channel=0)
measurement_dict_2013_100_LM_ch3 = dict(orientation='LM', threshold=100, year=2013, hdf5_path=hdf5_path, sigma=1.0,
                                        channel=1)
measurement_dict_2007_120_LM_ch2 = dict(orientation='LM', threshold=120, year=2007, hdf5_path=hdf5_path, sigma=1.0,
                                        channel=0)
measurement_dict_2004_140_LM_1_ch2 = dict(orientation='LM', threshold=140, year=2004, hdf5_path=hdf5_path,
                                          sigma=0.1, channel=0)

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
data_dicts = [measurement_dict_2020_120_LM_ch3, measurement_dict_2020_120_LM_ch4, measurement_dict_2004_140_LM_1_ch2]
measurement_dict = dict(orientation='PA', threshold=120, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=0)
n  = 50
intensitiy = np.repeat([250], n)
theta = np.repeat([0], n)
fraction_ex = np.repeat([0.6], n)
fraction_gaba_a = np.repeat([0.95], n)
fraction_nmda = np.repeat([0.61], n)
voltage_sigma = np.repeat([6], n)
init_pdf_sigma = np.repeat([2.5], n)
init_pdf_offset = np.repeat([0], n)


parameters = {'intensity': intensitiy, 'fraction_nmda': fraction_nmda, 'fraction_gaba_a': fraction_gaba_a,
              'fraction_ex': fraction_ex, 'theta': theta, 'enable_high_pass': True, 'min_delay': 5,
              'test_signal_from_file': True, 'i_scale': 5.148136e-9*3, 'detrend': False, 'plot_detrend': False,
              'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'mind_delay': 0,
              'delay_signal':True, 'delay': 0.9, 'error_mode': 'true',
              'file_args': measurement_dict_2020_120_PA_ch3, 'computation': 'vec',
              'nmm_parameters': {'connectivity_matrix': np.array([[0]]),
                                    'tau_ref': [0], #1.5
                                    'tau_mem': [12],
                                    'static_noise': True,
                                    'init_pdf_offset': init_pdf_offset,
                                    'init_pdf_sigma': init_pdf_sigma,
                                    'delay_kernel_type': 'alpha',
                                    'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3},
                                    'dv': dv,
                                    'dt': dt,
                                    'solver': 'Hu-2021',
                                    'voltage_sigma': voltage_sigma,
                                    'verbose': 1}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)

parameters_2 = parameters.copy()
param_update = dict(intensity=intensitiy[0], fraction_nmda=fraction_nmda[0], fraction_gaba_a=fraction_gaba_a[0],
                    fraction_ex=fraction_ex[0], theta=theta[0], computation='ser',
                    nykamp_parameters=dict(init_pdf_sigma=init_pdf_sigma[0], init_pdf_offset=init_pdf_offset[0]))
parameters_2.update(param_update)
di_model_ser = DI_wave_simulation(parameters=parameters_2, logname=None)
di_model_ser.simulate()
di_model.simulate()

vec_signal_1 = di_model.nmm_potentials[0]
vec_signal_2 = di_model.nmm_potentials[0]
ser_signal = di_model.mass_model_v_out
print(f'nrmse vec_sig 1 and ref {nrmse(vec_signal_1, ser_signal)}')
print(f'nrmse vec_sig 2 and ref {nrmse(vec_signal_2, ser_signal)}')
di_model.labelsize=17
# di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.0018, animate=False, savefig=save_figs)
di_model.plot_validation(fixed_ylim=False, save_fig=save_figs, labels=['Population Model', 'Measurement'], set_idx=0)
di_model.plot_validation(fixed_ylim=False, save_fig=save_figs, labels=['Population Model', 'Measurement'], set_idx=-1)
di_model.mass_model.clean()

# timing
# import time
#
# start = time.perf_counter()
#
# result = expensive_function()
#
# print(f"{time.perf_counter() - start:.6f} s")