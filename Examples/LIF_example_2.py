import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
from Model.LIF import LIF_population
matplotlib.use('TkAgg')

def v0(t):
    v0_bar = 700
    f = 10
    return v0_bar * (1 + np.sin(2*np.pi*f*t/1000))

def input_step_function(t):
    res = np.zeros_like(t)
    amp = 1.2e2
    res[(t>10) & (t<80)] = amp
    return res

T = 100
dt = 0.1
t = np.arange(0.0, T, dt)
# in_sine = input_sine_function(t)

dim = 100
# w0 = 3 * (10/dim)
w0 = 30/dim
con = w0*(np.ones((dim, dim)) - np.eye(dim))

con_prob = 0.2
for i in range(dim):
    no_connections = np.random.choice(np.where(con[i, :] != 0)[0], int(dim*con_prob))
    con[i, no_connections] = 0
# con = w0*np.random.uniform(size=(dim, dim))
# np.fill_diagonal(con, 0)
lif = LIF_population(T=T, tau_m=20, tref=3,  weights=con, n_neurons=dim, Iext=np.zeros_like(t), verbose=0)
lif.gen_poisson_spikes_input(rate=v0, i_max=15, delay=True)
lif.read_poisson_spikes_input(scale=1)
# lif.run()

# Visualize
# lif.plot_volt_trace(idx=3)
# lif.plot_volt_trace(idx=53)
lif.raster_plot()
# # times = [500, 1000, 2000, 3000, 4000]
# times = [100, 200, 300, 400, 500]
# lif.plot_voltage_hist(times=times)
# neuron_num = [0, 2, 5, 12, 22]
# lif.plot_firing_rate(bin_size=20, smoothing=True)
lif.plot_populations(bins=1000, smoothing=True, sigma=5, hide_refractory=True, cutoff=20)

# print(f'neuron 1 spikes: {lif.rec_spikes[0].shape}')
# print(f'neuron 2 spikes: {lif.rec_spikes[1].shape}')

