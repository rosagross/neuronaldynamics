import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
from Model.Nykamp_Model import Nykamp_Model_1
matplotlib.use('TkAgg')

plot_nykamp_basic = True
plot_di_model = False

dt = 0.01
dv = 0.001
T = 20
t = np.arange(0, T, dt)
Nt = t.shape[0]

if plot_nykamp_basic:

    a = 5e-5
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
    model_parameters['solver'] = 'Hu-2021'
    model_parameters['verbose'] = 1

    nyk1D = Nykamp_Model_1(parameters=model_parameters, name='Hu_solver_test')
    nyk1D.simulate()
    nyk1D.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.15, animate=False, savefig=False)
    nyk1D.clean()

if plot_di_model:
    # fn_session = '/home/erik/Downloads/gpc.pkl'
    fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
    # fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
    simulation_name = 'diw_2025_11_17_1'
    parameters = {'intensity': 304, 'fraction_nmda': 0.72, 'fraction_gaba_a': 0.91, 'fraction_ex': 0.68, 'plot_align': False,
                  'test_func_intensity': 2.5, 'test_func_t0': 0.35,
                  'test_signal_from_file': False,
                  'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt,
                  'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
                                        'tau_ref': [0], #1.5
                                        'tau_mem': [12],
                                        'input_type': 'current',
                                        'init_pdf_sigma': 0.1,
                                        'delay_kernel_type': 'alpha',
                                        'dv': dv}}
    di_model = DI_wave_simulation(parameters=parameters, logname=None)
    di_model.simulate()
    di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.2, animate=False, savefig=True)
    rhos = di_model.mass_model.rho
    # change in rho area
    drho = np.sum(rhos[0, :, 5]) - np.sum(rhos[0, :, -1])
    print(f"change in rho: {drho}")
    print(f'rho end: {np.sum(rhos[0, :, -1])}')
    di_model.plot_input_current()
    di_model.plot_validation()
    di_model.mass_model.clean()