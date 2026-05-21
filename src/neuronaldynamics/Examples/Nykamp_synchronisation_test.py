import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.Nykamp_Model import Nykamp_Model_1
matplotlib.use('TkAgg')



# init parameters
def step(t, t0=0, t1=80):
    res = 4e-4*np.ones_like(t)
    res[t < t0] = 0
    res[t > t1] = 0
    return res



### single population test ###
pars_1D = {}
pars_1D['connectivity_matrix'] = 30*np.array([[0]])
pars_1D['u_rest'] = -65
pars_1D['u_reset'] = -65
pars_1D['u_thr'] = -55
pars_1D['u_exc'] = 0
pars_1D['u_inh'] = -70
pars_1D['tau_mem'] = np.array([20])
pars_1D['tau_ref'] = np.array([0.6])
pars_1D['mu_gamma'] = np.array([[0.008, 0.027]])
pars_1D['var_coeff_gamma'] = 0.5*np.ones((1, 2))
pars_1D['tau_alpha'] = 1/3
pars_1D['n_alpha'] = 9
pars_1D['input_function'] = step
pars_1D['input_type'] = 'current'
pars_1D['input_function_type'] = 'custom'
pars_1D['input_function_idx'] = [0, 0]
pars_1D['population_type'] = ['exc']

T = 100  # 200
dt = 0.02 # 0.1
dv = 0.1
pars_1D['T'] = T
pars_1D['dt'] = dt
pars_1D['dv'] = dv

nyk1D = Nykamp_Model_1(parameters=pars_1D, name='nykamp_test_1D')
nyk1D.simulate()
nyk1D.plot(heat_map=True, plot_input=True)
nyk1D.clean()