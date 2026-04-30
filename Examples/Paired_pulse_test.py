import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
from Model.Nykamp_Model import Nykamp_Model_1
from Utils import get_peak_values
matplotlib.use('TkAgg')

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]

plot_nykamp_basic = False
plot_di_model = True
save_figs = False

dt = 0.01
dv = 0.01
# dt = 0.01, dv=0.05, very little noise (diff coeff 1.03)
# dt = 0.1, dv=0.5, completely noisy (diff coeff 1.1)
#TODO: find out about artificial oscillations in pdes depending on step size
# below 0.01, 0.01 setting (i.e. 0.01, 0.005) the threshold seems to be too large in E
# maybe inhibition idea is still the best
T = 50

t = np.arange(0, T, dt)
Nt = t.shape[0]

if plot_nykamp_basic:

    a = 0.1
    input_function =  a * ((2 * np.ones(Nt) + 1.5 * np.sin(t / 3)) + np.exp(-(t - 1.2) ** 2 / 0.1))

    model_parameters = {}
    model_parameters['connectivity_matrix'] = 0 * np.array([[1 / 2]])
    model_parameters['input_function'] = input_function
    model_parameters['tau_mem'] = [12]
    model_parameters['tau_ref'] = [0]
    model_parameters['input_type'] = 'stochastic-current'
    model_parameters['current_sigma'] = 0.1
    model_parameters['T'] = T
    model_parameters['dt'] = dt
    model_parameters['dv'] = dv
    model_parameters['init_pdf_sigma'] = 3
    model_parameters['init_pdf_offset'] = 6
    model_parameters['init_pdf_weight'] = 0.1
    model_parameters['solver'] = 'Hu-2021'
    model_parameters['verbose'] = 1

    nyk1D = Nykamp_Model_1(parameters=model_parameters, name='Hu_solver_long')
    nyk1D.simulate()
    nyk1D.plot(heat_map=True, plot_input=True, crop_rate=True, z_limit=0.001, animate=False, savefig=False)
    nyk1D.clean()
    rhos = nyk1D.rho
    # change in rho area
    drho = np.sum(rhos[0, :, 5]) - np.sum(rhos[0, :, -1])
    print(f"change in rho: {drho}")
    print(f'rho end: {np.sum(rhos[0, :, -1])}')

    u = rhos[0].T
    u_rest = nyk1D.u_rest
    x = nyk1D.v
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, u[-1])
    ax.plot(x, u[0], color='k', alpha=0.6)
    ax.set_xlabel('x')
    ax.set_xlabel('u(x)')
    ax.set_title(f'distribution at t= {T}')
    ax.grid()
    ax.vlines(x=u_rest, ymax=u[-1].max(), ymin=u[-1].min(), linestyle='--', color='k')
    ax.text(u_rest + 0.03, 0.03, 'x_rest', fontsize=12, transform=ax.transAxes)
    ax.text(u_rest - 0.1, 0.93, 'u_0', fontsize=12, transform=ax.transAxes)
    ax.set_ylim((u[-1].min(), u[-1].max() * 1.2))
    plt.show()

if plot_di_model:
    fn_session = '/home/erik/Downloads/gpc.pkl'
    # fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
    # fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
    # hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
    # hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data_detrended.hdf5"
    # hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
    hdf5_path = "/home/erik/Nextcloud_Uni/TMS Neuro Projects/M1_modeling/DI_wave_data/extracted_DI_waves/DiLazarro_di_wave_data.hdf5"

    simulation_name = '26_04_30_test'

    measurement_dict = dict(orientation='PA', threshold=120, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=0)
    parameters = {'intensity': 220, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.6, 'plot_align': False,
                  'test_func_intensity': 2.0, 'test_func_t0': 0.35, 'enable_high_pass': True, 'min_delay': 5,
                  'test_signal_from_file': False, 'i_scale': 5.148136e-9, 'detrend': False, 'plot_detrend': False,
                  'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'mind_delay': 0,
                  'theta': 0, 'delay_signal':True, 'delay': 0.9, 'error_mode': 'true',
                  'paired_pulse': True, 'pp_inpterval': 20,
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
                                        'current_sigma': 1.0,
                                        'verbose': 1}}

    di_model = DI_wave_simulation(parameters=parameters, logname=None)
    di_model.plot_input_current(savefig=save_figs)
    di_model.simulate()
    di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.0018, animate=False, savefig=save_figs)
    di_model.labelsize=20

    voltage_signal = di_model.mass_model_v_out
    di_model.labelsize=17
    di_model.plot_validation(fixed_ylim=False, save_fig=save_figs, labels=['Population Model', 'Measurement'])
    di_model.mass_model.clean()