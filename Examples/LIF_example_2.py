import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
from Model.LIF import LIF_population
matplotlib.use('TkAgg')

w0 = 30
dim = 100
# con = w0*(np.ones((dim, dim)) - np.eye(dim))
con = w0*np.random.uniform(size=(dim, dim))
np.fill_diagonal(con, 0)

lif = LIF_population(T=500, weights=con, n_neurons=dim)
lif.gen_poisson_spikes_input(rate=5, i_max=1000)
lif.run()

# Visualize
lif.plot_volt_trace(idx=0)
lif.plot_volt_trace(idx=1)
fig = plt.figure(figsize=(8, 8))
times = [500, 1000, 2000, 3000, 4000]

for n, time in enumerate(times):
  ax = fig.add_subplot(len(times), 1, int(n+1))
  ax.hist(lif.v[:, time], bins=100, density=True, alpha=0.7)

plt.tight_layout()
ax.set_xlabel('V in mv')
plt.show()

fig = plt.figure(figsize=(8, 8))
neuron_num = [0, 2, 5, 12, 22]

#TODO: find out how to make this plot the same way it is in the paper
# also create a spike raster plot and check for availabilty of this via pyrates
for n, n_neuron in enumerate(neuron_num):
  ax = fig.add_subplot(len(times), 1, int(n+1))
  # ax.hist(r[n_neuron, :], bins=100, density=True, alpha=0.7)
  ax.plot(np.mean(lif.r, axis=0))
  ax.set_ylabel('r in Hz')
plt.tight_layout()
ax.set_xlabel('time in ms')
plt.show()
# plt.subplots_adjust(hspace=0.5)

# plt.hist(v[:, 3000], bins=100, density=True, alpha=0.7)


print(f'neuron 1 spikes: {lif.rec_spikes[0].shape}')
print(f'neuron 2 spikes: {lif.rec_spikes[1].shape}')

