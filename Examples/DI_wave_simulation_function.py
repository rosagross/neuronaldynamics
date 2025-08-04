import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_2025_08_04_opt_test'
# [380.11439606   0.69919058   0.9749796    0.40482059  12.21559531]
parameters = {'intensity': 380.1, 'fraction_nmda': 0.7, 'fraction_gaba_a': 0.975, 'fraction_ex': 0.4, 'plot_align': False,
              'fn_session': fn_session, 'T': 8, 'name': simulation_name,
              'nykamp_parameters': {'connectivity_matrix': np.array([[12]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current'}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.simulate()
di_model.mass_model.plot(heat_map=True, plot_input=True)
di_model.plot_validation()
di_model.save_log(plot=True)
