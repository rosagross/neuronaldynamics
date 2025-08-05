import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_2025_08_04_opt_test'
# [109.61367607   0.48166361   0.92450167   0.50431263   9.15847855]
parameters = {'intensity': 119, 'fraction_nmda': 0.25, 'fraction_gaba_a': 0.97, 'fraction_ex': 0.58, 'plot_align': False,
              'fn_session': fn_session, 'T': 8, 'name': simulation_name,
              'nykamp_parameters': {'connectivity_matrix': np.array([[17]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current'}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.simulate()
di_model.mass_model.plot(heat_map=True, plot_input=True)
di_model.plot_validation()
di_model.save_log(plot=True)
