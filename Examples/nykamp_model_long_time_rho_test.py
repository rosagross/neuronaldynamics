import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
from Model.Nykamp_Model import Nykamp_Model_1
matplotlib.use('TkAgg')

plot_rate_model = False
plot_di_model = True

if plot_di_model:

    # fn_session = '/home/erik/Downloads/gpc.pkl'
    fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
    # fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
    simulation_name = 'diw_2025_11_10_4'
    parameters = {'intensity': 304, 'fraction_nmda': 0.72, 'fraction_gaba_a': 0.91, 'fraction_ex': 0.68, 'plot_align': False,
                  'test_func_intensity': 2.5, 'test_func_t0': 0.35,
                  'test_signal_from_file': True,
                  'fn_session': fn_session, 'T': 5, 'name': simulation_name, 'dt': 0.01, 'enable_high_pass': False,
                  'nykamp_parameters': {'connectivity_matrix': np.array([[0]]), # 17.75
                                        'tau_ref': [0], #1.5
                                        'tau_mem': [12],
                                        'input_type': 'current',
                                        'init_pdf_sigma': 1.0,
                                        'delay_kernel_type': 'alpha',
                                        'dv': 0.1,
                                        'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3}},
                  'g_eext_factor': 1,
                  'c_eext2_factor': 1,
                  'c_eext1_factor': 1}
    di_model = DI_wave_simulation(parameters=parameters, logname=None)
    di_model.simulate()
    di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.002, animate=True, savefig=True)
    rhos = di_model.mass_model.rho
    # change in rho area
    drho = np.sum(rhos[0, :, 5]) - np.sum(rhos[0, :, -1])
    print(f"change in rho: {drho}")
    print(f'rho end: {np.sum(rhos[0, :, -1])}')
    # di_model.plot_input_current()
    # di_model.plot_validation()
    di_model.mass_model.clean()


############################################################################################
# PLOT NYKAMP RATE VERSION                                                                 #
############################################################################################
if plot_rate_model:
    def step_population(t):
        t1 = 0.2
        t2 = 9.5
        res = np.zeros_like(t)
        res[t > t1] = 5
        res[t > t2] = 0
        return res

    pars_1D = {}
    pars_1D['connectivity_matrix'] = 0*np.array([[1/2]])

    # pars_1D['input_function'] = step_population
    pars_1D['input_function'] = step_population
    pars_1D['input_function_type'] = 'custom'
    pars_1D['input_function_idx'] = [0, 0]
    pars_1D['population_type'] = ['exc']
    pars_1D['tau_mem'] = [12]
    pars_1D['tau_ref'] = [0]


    # pars_1D['input_type'] = 'current'
    pars_1D['c_mem'] = [0.2]  # 0.2F capacitance

    dt = 0.01 # 0.1
    dv = 0.1
    pars_1D['T'] = 10
    pars_1D['dt'] = dt
    pars_1D['dv'] = dv


    nyk1D = Nykamp_Model_1(parameters=pars_1D, name='diw_step_func_rate_test_3')
    nyk1D.simulate()
    nyk1D.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.15, animate=False, savefig=False)
    nyk1D.clean()


    rhos = nyk1D.rho
    # change in rho area
    drho = np.sum(rhos[0, :, 5]) - np.sum(rhos[0, :, -1])
    print(f"change in rho rate model: {drho}")
    print(f'rho end: {np.sum(rhos[0, :, -1])}')