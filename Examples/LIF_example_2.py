import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
from Model.LIF import LIF_population
matplotlib.use('TkAgg')

w0 = 30
dim = 10
# con = w0*(np.ones((dim, dim)) - np.eye(dim))
con = w0*np.random.uniform(size=(dim, dim))
np.fill_diagonal(con, 0)

lif = LIF_population(T=100, tau_m=20,  weights=con, n_neurons=dim)
lif.gen_poisson_spikes_input(rate=0.1, i_max=2e4)
lif.run()

# Visualize
lif.plot_volt_trace(idx=0)
lif.raster_plot()
times = [500, 1000, 2000, 3000, 4000]
lif.plot_voltage_hist(times=times)
neuron_num = [0, 2, 5, 12, 22]
lif.plot_firing_rate(neuron_num=neuron_num)

# plt.subplots_adjust(hspace=0.5)

# plt.hist(v[:, 3000], bins=100, density=True, alpha=0.7)





print(f'neuron 1 spikes: {lif.rec_spikes[0].shape}')
print(f'neuron 2 spikes: {lif.rec_spikes[1].shape}')

