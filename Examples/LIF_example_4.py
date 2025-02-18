import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import random
from tqdm import tqdm
from Model.LIF import Conductance_LIF
matplotlib.use('TkAgg')

def v0(t):
    v0_bar = 0.7  # 700 spikes / second / 1000 to convert to ms
    f = 10
    return v0_bar * (1 + np.sin(2*np.pi*f*t/1000))

def input_step_function(t):
    res = np.zeros_like(t)
    amp = 1.2e2
    res[(t>10) & (t<80)] = amp
    return res

T = 100
dt = .1
t = np.arange(0.0, T, dt)

dim = 1000
con = np.zeros((dim, dim))
n_connection = int(dim/10)

for i in tqdm(range(dim), f'computing random neuron connections for {dim} neurons'):
    possible_connections = np.arange(dim)
    # exclude current idx
    possible_connections = possible_connections[possible_connections!=i]
    connections_i = random.sample(possible_connections.tolist(), n_connection)
    if np.unique(np.array(connections_i)).shape[0] != n_connection:
        print(f"failed sampling!!!, idx {i}")
    con[i, connections_i] = 1

neuron_parameters = {'T': T, 'tau_m': 20, 't_ref': 3, 'weights': con, 'n_neurons': dim}
lif = Conductance_LIF(parameters=neuron_parameters)
lif.gen_poisson_spikes_input(rate=v0, i_max=1, delay=False)
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
lif.plot_populations(bins=1000, smoothing=True, sigma=5, hide_refractory=True, cutoff=None)

# print(f'neuron 1 spikes: {lif.rec_spikes[0].shape}')
# print(f'neuron 2 spikes: {lif.rec_spikes[1].shape}')

