import numpy as np
import matplotlib
from Model.DI_wave import DI_wave_simulation
matplotlib.use('TkAgg')

# uncomment for Times new roman font
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]

plot_nykamp_basic = False
save_figs = False

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
# hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data_detrended.hdf5"

simulation_name = '26_04_01_test'

measurement_dict_2020_140_PA_ch3 = dict(orientation='PA', threshold=140, year=2020, hdf5_path=hdf5_path, sigma=1.0)


measurement_dict = dict(orientation='PA', threshold=120, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=0)
parameters = {'intensity': 250, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.6, 'plot_align': False,
              'test_func_intensity': 2.0, 'test_func_t0': 0.35, 'enable_high_pass': False, 'min_delay': 5,
              'test_signal_from_file': True, 'i_scale': 5.148136e-6, 'detrend': True, 'plot_detrend': False,
              'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'mind_delay': 0,
              'theta': 90, 'delay_signal':True,
              'file_args': measurement_dict_2020_140_PA_ch3,
              'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
                                    'tau_ref': [0], #1.5
                                    'tau_mem': [12],
                                    'input_type': 'stochastic-current',
                                    'static_noise': True,
                                    'init_pdf_offset': 0,
                                    'init_pdf_sigma': 0.1,
                                    'init_pdf_weight': 0,
                                    'delay_kernel_type': 'alpha',
                                    'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3},
                                    'dv': dv,
                                    'dt': dt,
                                    'solver': 'Hu-2021',
                                    'current_sigma': 12,
                                    'verbose': 1}}


di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.simulate()

di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.0018, animate=False, savefig=save_figs)
di_model.labelsize=20
di_model.plot_input_current(savefig=save_figs)
voltage_signal = di_model.mass_model_v_out

di_model.labelsize=17
di_model.plot_validation(fixed_ylim=True, save_fig=save_figs, labels=['Population Model', 'Measurement'])
di_model.mass_model.clean()

# extra PA, LM examples
# parameters['theta'] = 30
# parameters['name'] = 'CNS_26_example_PA'
# di_model = DI_wave_simulation(parameters=parameters)
# di_model.simulate()
# di_model.plot_voltage(savefig=save_figs)
# di_model.mass_model.clean()
# parameters['theta'] = 150
# parameters['name'] = 'CNS_26_example_LM'
# di_model = DI_wave_simulation(parameters=parameters)
# di_model.simulate()
# di_model.plot_voltage(savefig=save_figs)
# di_model.mass_model.clean()