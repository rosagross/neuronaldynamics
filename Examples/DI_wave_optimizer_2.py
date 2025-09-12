import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_opt_25_09_11_2'
parameters = {'intensity': 220, 'fraction_nmda': 0.5, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.4, 'plot_align': False,
              'test_func_intensity': 2.5, 'test_func_t0': 0.25,
              'test_signal_from_file': True,
              'fn_session': fn_session, 'T': 10, 'name': simulation_name, 'dt': 0.02, 'enable_high_pass': False,
              'nykamp_parameters': {'connectivity_matrix': np.array([[10]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current'}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)


model_parameters = ['intensity', 'fraction_nmda', 'fraction_gaba_a', 'fraction_ex', 'mass_model_connectivity_matrix']
model_parameter_bounds = [[150, 400], [0.25, 0.75], [0.9, 1.0], [0.2, 0.8], [0, 50]]
opt_parameters = {'optimizer': 'hierarchical', 'eps': 0.05, 'max_iter': 5, 'n_grid': 50,
                  'model_parameters': model_parameters, 'bounds': model_parameter_bounds, 'x_out': 'mass_model_v_out',
                  'fn_session': fn_session, 'T': 10, 'nykamp_parameters': {'tqdm_disable': True}, 'dt': 0.02,
                  'enable_high_pass': False}


di_model.optimize(opt_params=opt_parameters)
opt_params = di_model.optimimization_algorithm.optimum
print(opt_params)
# this is not really working, check updating te parameters
# opt_param_dict = {'intensity': opt_params[0], 'fraction_nmda': opt_params[1], 'fraction_gaba_a': opt_params[2],
#                   'fraction_ex': opt_params[3], 'fn_session': fn_session, 'T': 8, 'name': simulation_name, 'dt': 0.01,
#                   'nykamp_parameters': {'connectivity_matrix': np.array([[opt_params[4]]]),
#                                        'tau_ref': [1.5],
#                                        'tau_mem': [12],
#                                        'input_type': 'current'},
#                   'enable_high_pass': False}
# di_model = DI_wave_simulation(parameters=parameters, logname=None)
# di_model.simulate()
# di_model.mass_model.plot(heat_map=True, plot_input=True)
# di_model.plot_validation()
# di_model.save_log(plot=True)