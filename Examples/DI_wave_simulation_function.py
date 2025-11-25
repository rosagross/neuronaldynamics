import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

# fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_2025_11_04_5'
# [140.34858132   0.67971667   0.98021345   0.43615625  13.24401196]
# [150   0.6983768    0.98613453   0.57663151  22.7918771 ]
# [204.51832028   0.72726794   0.90854288   0.67587172  17.75456614]
# best param set [200.7025335    0.75         0.91448355   0.53258413  19.87193974] with error: 0.06024330293723818
# [2.80882935e+02 7.19944251e-01 9.31411365e-01 2.03155545e-01 2.17894470e+01] # real data special case
# [3.22442644e+02 3.03558687e-01 9.41387554e-01 7.99044186e-01, 1.32660674e+01] # hierarchical optimium from g_eext*100 example
parameters = {'intensity': 322, 'fraction_nmda': 0.30, 'fraction_gaba_a': 0.94, 'fraction_ex': 0.80, 'plot_align': False,
              'test_func_intensity': 2.5, 'test_func_t0': 0.35,
              'test_signal_from_file': True,
              'fn_session': fn_session, 'T': 10, 'name': simulation_name, 'dt': 0.01, 'enable_high_pass': False,
              'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
                                    'tau_ref': [0],
                                    'tau_mem': [12],
                                    'input_type': 'current',
                                    'init_pdf_sigma': 1.0},
              'g_eext_factor': 1,
              'c_eext2_factor': 1}
di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.simulate()
di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.15, animate=False, savefig=False)
# di_model.plot_input_current()
# di_model.get_test_signal(plot=True)
# di_model.plot_convolution()
di_model.plot_validation()
# di_model.save_log(plot=True)
di_model.mass_model.clean()
