import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
simulation_name = 'diw_2025_10_28_07'
parameters = {'intensity': 204, 'fraction_nmda': 0.72, 'fraction_gaba_a': 0.91, 'fraction_ex': 0.68, 'plot_align': False,
              'test_func_intensity': 2.5, 'test_func_t0': 0.35,
              'test_signal_from_file': True,
              'fn_session': fn_session, 'T': 20, 'name': simulation_name, 'dt': 0.02, 'enable_high_pass': False,
              'nykamp_parameters': {'connectivity_matrix': np.array([[17.75]]),
                                    'tau_ref': [1.5],
                                    'tau_mem': [12],
                                    'input_type': 'current',
                                    'init_pdf_sigma': 1.0,
                                    'delay_kernel_type': 'alpha',
                                    'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3}}}
di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.simulate()
di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.15, animate=True)
rhos = di_model.mass_model.rho
# change in rho area
drho = np.sum(rhos[0, :, 5]) - np.sum(rhos[0, :, -1])
print(f"change in rho: {drho}")
# di_model.plot_input_current()
di_model.plot_validation()
di_model.mass_model.clean()
