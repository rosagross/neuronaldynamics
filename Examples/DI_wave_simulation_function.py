import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

# fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_2025_09_22_02'
# [140.34858132   0.67971667   0.98021345   0.43615625  13.24401196]
# [150   0.6983768    0.98613453   0.57663151  22.7918771 ]
# [2.80882935e+02 7.19944251e-01 9.31411365e-01 2.03155545e-01 2.17894470e+01] # real data special case
parameters = {'intensity': 387, 'fraction_nmda': 0.38, 'fraction_gaba_a': 0.92, 'fraction_ex': 0.77, 'plot_align': False,
              'test_func_intensity': 2.5, 'test_func_t0': 0.35,
              'test_signal_from_file': True,
              'fn_session': fn_session, 'T': 10, 'name': simulation_name, 'dt': 0.02, 'enable_high_pass': False,
              'nykamp_parameters': {'connectivity_matrix': np.array([[17.3]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current',
                                    'init_pdf_sigma': 1.0}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.simulate()
di_model.mass_model.plot(heat_map=True, plot_input=False, plot_combined=False, z_limit=0.15)
# di_model.plot_input_current()
# di_model.get_test_signal(plot=True)
# di_model.plot_convolution()
di_model.plot_validation()
# di_model.save_log(plot=True)
di_model.mass_model.clean()
