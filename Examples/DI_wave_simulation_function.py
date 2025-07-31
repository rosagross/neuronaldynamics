import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'di_wave_test_8'
parameters = {'intensity': 200, 'fraction_nmda': 0.5, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.8, 'plot_align': False,
              'fn_session': fn_session, 'T': 30, 'name': simulation_name,
              'nykamp_parameters': {'connectivity_matrix': np.array([[15]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current'}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.simulate()
# di_model.mass_model.plot(heat_map=True, plot_input=True)
di_model.plot_validation()
di_model.save_log(plot=True)
