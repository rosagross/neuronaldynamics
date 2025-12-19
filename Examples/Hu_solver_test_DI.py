import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
from Model.Nykamp_Model import Nykamp_Model_1
matplotlib.use('TkAgg')

plot_nykamp_basic = False
plot_di_model = True

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]

if plot_nykamp_basic:

    a = 0.1
    input_function =  a * ((2 * np.ones(Nt) + 1.5 * np.sin(t / 3)) + np.exp(-(t - 1.2) ** 2 / 0.1))
    # t_off = 10
    # t_again = 40
    # off_idx = np.where(t > t_off)
    ##  reactivation for T=50
    # again_idx = np.where(t > t_again)[0][0]
    # input_function[off_idx] = 0
    # input_function[again_idx:] = input_function[0:input_function.shape[0] - again_idx]

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
    # fn_session = '/home/erik/Downloads/gpc.pkl'
    fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
    # fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
    simulation_name = 'diw_Hu_solver_test'

    # [3.80187468e+02 6.24947346e-01 9.15432571e-01 6.77733943e-01
    #  5.89647134e-03 2.89137701e-01 3.55730239e+00 1.71540732e-06
    #  8.49486711e+01]
    # [3.09220226e+02 5.94160306e-01 9.85183116e-01 6.45454831e-01
    #  1.13762176e+00 4.96923672e-01 5.27521518e-01 1.29366546e-06
    #  5.73742344e+01]
    # [2.31857700e+02 6.12769298e-01 9.44351120e-01 5.71123641e-01
    #  9.93984614e+00 7.03567602e-01 7.27529179e+00 1.05448267e-06
    #  2.05980423e+01 2.98320822e+02]
    # ['intensity', 'fraction_nmda', 'fraction_gaba_a', 'fraction_ex',
    # 'pdf_offset', 'pdf_sigma', 'pdf_weight', 'i_scale',
    #  'current_sigma']
    parameters = {'intensity': 250, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.94, 'fraction_ex': 0.57, 'plot_align': False,
                  'test_func_intensity': 1.5, 'test_func_t0': 0.35, 'enable_high_pass': True,
                  'test_signal_from_file': False, 'i_scale': 5e-07,
                  'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt,
                  'nykamp_parameters': {'connectivity_matrix': np.array([[1400]]),
                                        'tau_ref': [0], #1.5
                                        'tau_mem': [12],
                                        'input_type': 'stochastic-current',
                                        'init_pdf_offset': 0,
                                        'init_pdf_sigma': 0.01,
                                        'init_pdf_weight': 20,
                                        'delay_kernel_type': 'alpha',
                                        'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3},
                                        'dv': dv,
                                        'dt': dt,
                                        'solver': 'Hu-2021',
                                        'current_sigma': 50,
                                        'verbose': 1}}

    di_model = DI_wave_simulation(parameters=parameters, logname=None)
    di_model.simulate()
    di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.001, animate=False, savefig=False)
    rhos = di_model.mass_model.rho
    # change in rho area
    drho = np.sum(rhos[0, :, 5]) - np.sum(rhos[0, :, -1])
    print(f"change in rho: {drho}")
    print(f'rho end: {np.sum(rhos[0, :, -1])}')
    di_model.plot_validation()
    di_model.mass_model.clean()
