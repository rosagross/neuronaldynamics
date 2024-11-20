import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
from Model.LIF import LIF_population
matplotlib.use('TkAgg')

def input_sine_function(t):
    v0 = .7e2 # .7
    f = 10
    return v0 * (1 + np.sin(2*np.pi*f*t/1000))

T = 100
dt = 0.1
t = np.arange(0.0, T, dt)
in_sine = input_sine_function(t)

w0 = 30
dim = 7000
con = w0*(np.ones((dim, dim)) - np.eye(dim))
# con = w0*np.random.uniform(size=(dim, dim))
# np.fill_diagonal(con, 0)

lif = LIF_population(T=T, tau_m=20,  weights=con, n_neurons=dim, Iext=in_sine, verbose=0)
lif.gen_poisson_spikes_input(rate=0.3, i_max=5e5) #input in nA?
lif.run()

# Visualize
lif.plot_volt_trace(idx=5003)
lif.plot_volt_trace(idx=5004)
lif.plot_volt_trace(idx=5005)
lif.raster_plot()
# times = [500, 1000, 2000, 3000, 4000]
times = [100, 200, 500, 700, 900]
lif.plot_voltage_hist(times=times)
neuron_num = [0, 2, 5, 12, 22]
# lif.plot_firing_rate(neuron_num=neuron_num)

print(f'neuron 1 spikes: {lif.rec_spikes[0].shape}')
print(f'neuron 2 spikes: {lif.rec_spikes[1].shape}')

