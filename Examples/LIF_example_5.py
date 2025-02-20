import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import random
from tqdm import tqdm
from Model.LIF import Conductance_LIF
matplotlib.use('TkAgg')

def v0(t):
    v0_bar = 700  # 700 spikes / second / 1000 to convert to ms
    f = 10
    return (v0_bar/1000) * (1 + np.sin(2*np.pi*f/1000*t))

T = 100
dt = .1
t = np.arange(0.0, T, dt)

dim = 2000
con = np.zeros((dim, dim))
w_bar = 190  # int(dim/10)
population_types = ['exc', 'inh']
neuron_types = np.concatenate((np.zeros(1000), np.ones(1000)))
population_connections = w_bar * np.array([[0.5, 1], [1, 1]])
coeff_of_var = 0.5 * np.ones((2, 2))
mu = np.array([[0.008, 0.027], [0.020, 0.066]])

neuron_parameters = {'T': T, 'tau_m': np.array([20, 10]), 't_ref': np.array([3, 1]),
                     'E_e_i': np.array([0, -70]), ''
                     'n_neurons': dim, 'population_type': population_types,
                     'population_weights': population_connections, 'mu': mu, 'coeff_of_var': coeff_of_var,
                     'type_mask': neuron_types}
lif = Conductance_LIF(parameters=neuron_parameters)
lif.compute_connections()
lif.gen_poisson_spikes_input(rate=v0, i_max=1, delay=False, input_type='ee', population=0)
# lif.gen_poisson_spikes_input(rate=v0, i_max=1, delay=False, input_type='ie')
# lif.read_poisson_spikes_input(scale=1)
lif.run()

# Visualize
lif.plot_volt_trace(idx=3)
# lif.plot_volt_trace(idx=53)
lif.raster_plot()
# # times = [500, 1000, 2000, 3000, 4000]
# times = [100, 200, 300, 400, 500]
# lif.plot_voltage_hist(times=times)
# neuron_num = [0, 2, 5, 12, 22]
# lif.plot_firing_rate(bin_size=20, smoothing=True)
lif.plot_populations(bins=1000, smoothing=True, sigma=12, hide_refractory=True, cutoff=None)

# print(f'neuron 1 spikes: {lif.rec_spikes[0].shape}')
# print(f'neuron 2 spikes: {lif.rec_spikes[1].shape}')

