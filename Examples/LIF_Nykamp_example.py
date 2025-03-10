import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import random
from tqdm import tqdm
from Model.LIF import Conductance_LIF
from Model.Nykamp_Model import Nykamp_Model_1
from Utils import compare_firing_rate
matplotlib.use('TkAgg')

########################################################################################################################
# LIF model #
########################################################################################################################

# set-up time and input model
def v0(t):
    v0_bar = 700  # 700 spikes / second / 1000 to convert to ms
    f = 10
    return (v0_bar/1000) * (1 + np.sin(2*np.pi*f/1000*t))

T = 100
dt = 0.1
t = np.arange(0.0, T, dt)

# set up model
dim = 1000
n_neurons = 3*dim
con = np.zeros((n_neurons, n_neurons))
w_bar = 610
population_types = ['exc', 'inh', 'exc']
neuron_types = np.concatenate((np.zeros(dim), np.ones(dim), int(2)*np.ones(dim)))
c_array = np.array([[1/2, 1, 1], [1, 1, 1], [1, 1, 1]])
population_connections = w_bar * c_array
coeff_of_var = 0.5 * np.ones((3, 3))
# set mu values for all populations, the exc populations keep the same mu values
mu = np.zeros((3, 3))
mu_ee = 0.008
mu_ei = 0.027
mu_ie = 0.020
mu_ii = 0.066
mu[(0, 0, 2, 2), (0, 2, 0, 2)] = mu_ee
mu[(0, 2), (1, 1)] = mu_ei
mu[(1, 1), (0, 2)] = mu_ie
mu[1, 1] = mu_ii
neuron_parameters = {'T': T, 'tau_m': np.array([20, 10, 20]), 't_ref': np.array([3, 1, 3]),
                     'E_e_i': np.array([0, -70, 0]), ''
                     'n_neurons': n_neurons, 'population_type': population_types,
                     'population_weights': population_connections, 'mu': mu, 'coeff_of_var': coeff_of_var,
                     'type_mask': neuron_types}

# transfer build LIF network and input time series
lif = Conductance_LIF(parameters=neuron_parameters)
lif.compute_connections()
lif.gen_poisson_spikes_input(rate=v0, i_max=1, delay=False, input_type='ee', population=0)

# run simulation
lif.run()

# visualize
lif.plot_volt_trace(idx=3, population_idx=2)
lif.raster_plot()
lif.plot_populations(bins=1000, smoothing=True, sigma=10, hide_refractory=True, cutoff=None, size=0.7)

########################################################################################################################
# Nykamp model #
########################################################################################################################

parameters_nykamp = {}
w0 = 30
parameters_nykamp['u_rest'] = -65
parameters_nykamp['u_thr'] = -55
parameters_nykamp['u_exc'] = 0
parameters_nykamp['u_inh'] = -70
parameters_nykamp['tau_alpha'] = 1/3
parameters_nykamp['n_alpha'] = 9
parameters_nykamp['input_function'] = v0
parameters_nykamp['input_function_type'] = 'custom'
parameters_nykamp['input_function_idx'] = [0, 0]
parameters_nykamp['connectivity_matrix'] = w0 * c_array
parameters_nykamp['tau_mem'] = np.array([20, 10, 20])
parameters_nykamp['tau_ref'] = np.array([3, 1, 3])
parameters_nykamp['mu_gamma'] = np.array([[0.008, 0.027], [0.020, 0.066], [0.008, 0.027]])
parameters_nykamp['var_coeff_gamma'] = 0.5*np.ones((3, 2))
parameters_nykamp['population_type'] = ['exc', 'inh', 'exc']

dv = 0.01

nyk = Nykamp_Model_1(parameters=parameters_nykamp, name='nykamp_test_3D')
nyk.simulate(T=T, dt=dt, dv=dv, verbose=0, sparse_mat=True)

# saving shenanigans #

tag = np.random.randint(0, 100, size=5)
tag = str(tag).replace('[', '')
tag = tag.replace(']', '')
tag = tag.replace(' ', '')
save_name = 'nyk_vs_lif_' + tag
compare_firing_rate('nykamp_test_3D', 'Conductance_LIF', smooth=False, idx=0, save_fname=save_name)
tag = np.random.randint(0, 100, size=5)
tag = str(tag).replace('[', '')
tag = tag.replace(']', '')
tag = tag.replace(' ', '')
save_name = 'nyk_vs_lif_' + tag
compare_firing_rate('nykamp_test_3D', 'Conductance_LIF', smooth=False, idx=1, save_fname=save_name)
tag = np.random.randint(0, 100, size=5)
tag = str(tag).replace('[', '')
tag = tag.replace(']', '')
tag = tag.replace(' ', '')
save_name = 'nyk_vs_lif_' + tag
compare_firing_rate('nykamp_test_3D', 'Conductance_LIF', smooth=True, idx=0, save_fname=save_name)
tag = np.random.randint(0, 100, size=5)
tag = str(tag).replace('[', '')
tag = tag.replace(']', '')
tag = tag.replace(' ', '')
save_name = 'nyk_vs_lif_' + tag
compare_firing_rate('nykamp_test_3D', 'Conductance_LIF', smooth=True, idx=1, save_fname=save_name)





