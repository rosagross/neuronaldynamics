import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')
T = 15
dt = 0.01
dv = 0.01
fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_opt_hu_12_12_25'
parameters = {'intensity': 220, 'fraction_nmda': 0.5, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.4, 'plot_align': False,
              'test_func_intensity': 2.5, 'test_func_t0': 0.25,
              'test_signal_from_file': True, 'i_scale': 5.148136e-9*300,
              'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'enable_high_pass': False,
              'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
                                    'tau_ref': [0],
                                    'tau_mem': [12],
                                    'dv': dv,
                                    'delay_kernel_type': 'alpha',
                                    'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3},
                                    'input_type': 'stochastic-current',
                                    'solver': 'hu-2021'}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)

# TODO: exclude C, include pdf weight, offset and sigma, use i_scale*350 value (or 300)
#  Maybe also just try to fit aaron model output to the shape in hu_solver_exmaple

model_parameters = ['intensity', 'fraction_nmda', 'fraction_gaba_a', 'fraction_ex', 'pdf_offset', 'pdf_sigma', 'pdf_weight']
model_parameter_bounds = [[200, 400], [0.25, 0.75], [0.9, 1.0], [0.5, 0.8], [0, 12], [0.01, 5], [0.05, 20]]
opt_parameters = {'optimizer': 'hierarchical', 'eps': 0.05, 'max_iter': 5, 'n_grid': 50,
                  'model_parameters': model_parameters, 'bounds': model_parameter_bounds, 'x_out': 'mass_model_v_out',
                  'fn_session': fn_session, 'T': T, 'nykamp_parameters': {'tqdm_disable': True}, 'dt': dt}


di_model.optimize(opt_params=opt_parameters)
opt_params = di_model.optimimization_algorithm.optimum
print(opt_params)
opt_parameters = {'intensity': opt_params[0], 'fraction_nmda': opt_params[1], 'fraction_gaba_a': opt_params[2],
                  'fraction_ex': opt_params[3], 'pdf_offset': opt_params[4], 'pdf_sigma' : opt_params[5],
                  'pdf_weight': opt_params[6], 'plot_align': False,
              'test_func_intensity': 2.5, 'test_func_t0': 0.25,
              'test_signal_from_file': True,
              'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'enable_high_pass': False,
              'nykamp_parameters': {'tau_ref': [0],
                                    'tau_mem': [12],
                                    'dv': dv,
                                    'delay_kernel_type': 'alpha',
                                    'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1 / 3},
                                    'input_type': 'stochastic-current',
                                    'solver': 'hu-2021'
                                    }}
di_model_2 = DI_wave_simulation(parameters=opt_parameters, logname=None)
di_model_2.simulate()
di_model_2.mass_model.plot()
di_model_2.plot_validation()