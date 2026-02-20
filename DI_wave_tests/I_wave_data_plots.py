import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
from Model.Nykamp_Model import Nykamp_Model_1
matplotlib.use('TkAgg')

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
# hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"

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
measurement_dict = dict(orientation='PA', threshold=120, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=0)
parameters = {'intensity': 250, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.5, 'plot_align': False,
              'test_func_intensity': 2.0, 'test_func_t0': 0.35, 'enable_high_pass': False, 'min_delay': 5,
              'test_signal_from_file': True, 'i_scale': 5.148136e-6, 'detrend': False, 'plot_detrend': False,
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

for i_dict, dict in enumerate(data_dicts):
    parameters['file_args'] = dict
    di_model = DI_wave_simulation(parameters=parameters, logname=None)
    di_model.get_test_signal(plot=True, from_file=True, hdf5_args=di_model.file_args, highpass=True)


