import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import random
from tqdm import tqdm
from Model.LIF import Conductance_LIF
matplotlib.use('TkAgg')

# set-up time and input model
def v0(t):
    v0_bar = 700  # 700 spikes / second / 1000 to convert to ms
    f = 10
    return (v0_bar/1000) * (1 + np.sin(2*np.pi*f/1000*t))

T = 100
dt = 1.0
t = np.arange(0.0, T, dt)

# set up model
dim = 1000
n_neurons = 2*dim
con = np.zeros((n_neurons, n_neurons))
w_bar = 300
population_types = ['exc', 'inh']
neuron_types = np.concatenate((np.zeros(dim), np.ones(dim)))
population_connections = w_bar * np.array([[0.5, 1], [1, 1]])
coeff_of_var = 0.5 * np.ones((2, 2))
mu = np.array([[0.008, 0.027], [0.020, 0.066]]).T
neuron_parameters = {'T': T, 'tau_m': np.array([20, 10]), 't_ref': np.array([3, 1]),
                     'E_e_i': np.array([0, -70]), ''
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
# lif.plot_volt_trace(idx=3, population_idx=1)
# lif.plot_volt_trace(idx=5, population_idx=1)
# lif.plot_volt_trace(idx=13, population_idx=1)
# lif.plot_volt_trace(idx=63, population_idx=1)
lif.raster_plot()
lif.plot_populations(bins=1000, smoothing=True, sigma=10, hide_refractory=True, cutoff=None)


