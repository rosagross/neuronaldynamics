import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Model.DI_wave import DI_wave_simulation
from Optimizers.Optimizer import Hierarchical_Random, GA

fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_opt_25_09_29_1'
parameters = {'intensity': 220, 'fraction_nmda': 0.5, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.4, 'plot_align': False,
              'test_func_intensity': 2.5, 'test_func_t0': 0.25,
              'test_signal_from_file': True,
              'fn_session': fn_session, 'T': 10, 'name': simulation_name, 'dt': 0.02, 'enable_high_pass': False,
              'nykamp_parameters': {'connectivity_matrix': np.array([[10]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current'}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.get_test_signal()


model_parameters = ['intensity', 'fraction_nmda', 'fraction_gaba_a', 'fraction_ex', 'mass_model_connectivity_matrix']
model_parameter_bounds = [[200, 400], [0.25, 0.75], [0.9, 1.0], [0.5, 0.8], [0, 50]]  # frac exc was 0.2 min
opt_parameters = {'optimizer': 'hierarchical', 'eps': 0.05, 'max_iter': 5, 'n_grid': 50,
                  'model_parameters': model_parameters, 'bounds': model_parameter_bounds, 'x_out': 'mass_model_v_out',
                  'fn_session': fn_session, 'T': 10, 'nykamp_parameters': {'tqdm_disable': True}, 'dt': 0.02,
                  'enable_high_pass': False}

opt_parameters['y'] = np.array([0])
opt_parameters['simulation_class'] = di_model
opt_parameters['simulate'] = di_model.simulate
opt_parameters['x_out'] = 'mass_model_v_out'
opt_parameters['reference'] = di_model.target
opt_parameters['n_iter'] = 1
opt_parameters['N1'] = 50
opt_parameters['tolerance'] = 0.01
opt_parameters['verbose'] = 1
optimizer = GA(parameters=opt_parameters)
# opt_params['max_iter'] = 1000
# optimizer = Hierarchical_Random(parameters=opt_params)
optimizer.run()
optimal_param = optimizer.optimum
optimizer.plot_fit()
np.savetxt('parameters.txt', optimizer.ps)
np.savetxt('errors.txt', optimizer.errors)
print(optimal_param)


opt_parameters = {'intensity': optimal_param[0], 'fraction_nmda': optimal_param[1], 'fraction_gaba_a': optimal_param[2],
                  'fraction_ex': optimal_param[3], 'plot_align': False,
              'test_func_intensity': 2.5, 'test_func_t0': 0.25,
              'test_signal_from_file': True,
              'fn_session': fn_session, 'T': 10, 'name': simulation_name, 'dt': 0.02, 'enable_high_pass': False,
              'nykamp_parameters': {'connectivity_matrix': np.array([[optimal_param[4]]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current'}}
di_model_2 = DI_wave_simulation(parameters=parameters, logname=None)
di_model_2.simulate()
di_model_2.mass_model.plot()
di_model_2.plot_validation()