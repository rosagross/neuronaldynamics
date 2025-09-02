import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_2025_08_13_12'
# [150   0.31881403   0.95170811   0.27486198  27.7887928 ]
parameters = {'intensity': 150, 'fraction_nmda': 0.31, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.27, 'plot_align': False,
              'fn_session': fn_session, 'T': 8, 'name': simulation_name, 'dt': 0.05, 'enable_high_pass': False,
              'nykamp_parameters': {'connectivity_matrix': np.array([[28]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current'}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.simulate()
# di_model.mass_model.plot(heat_map=True, plot_input=True)
# di_model.plot_convolution()
di_model.plot_validation()
di_model.save_log(plot=True)
di_model.mass_model.clean()
