import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation

matplotlib.use('TkAgg')

fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'

parameters = {'intensity': 150, 'fraction_nmda': 0.5, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.4, 'plot_align': True,
              'fn_session': fn_session, 'nykamp_parameters':{'connectivity_matrix': np.array([[15]]), 'tau_ref':[1.2],
                                                             'tau_mem':[12]}}
di_model = DI_wave_simulation(parameters=parameters)
di_model.simulate()
di_model.mass_model.plot(heat_map=True, plot_input=True)
di_model.plot_validation()