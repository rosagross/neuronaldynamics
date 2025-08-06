import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_2025_08_06_4'
# [111.39759581   0.70269888   0.96519183   0.67856609   6.65007605]
parameters = {'intensity': 111, 'fraction_nmda': 0.70, 'fraction_gaba_a': 0.96, 'fraction_ex': 0.67, 'plot_align': False,
              'fn_session': fn_session, 'T': 15, 'name': simulation_name, 'dt': 0.05,
              'nykamp_parameters': {'connectivity_matrix': np.array([[6.65]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current'}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.simulate()
di_model.mass_model.plot(heat_map=True, plot_input=True)
di_model.plot_validation()
di_model.save_log(plot=True)
