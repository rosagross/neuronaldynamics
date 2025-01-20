import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
from Model.LIF import LIF_population
matplotlib.use('TkAgg')

def input_step_function(t):
    res = np.zeros_like(t)
    amp = 5e2
    res[(t>10) & (t<30)] = amp
    return res

T = 100
dt = 0.1
t = np.arange(0.0, T, dt)

dim = 80
w0 = 5 * (40/dim) # scaling is done here as to preserve input strength
# con = w0*(np.ones((dim, dim)) - np.eye(dim))
# con = w0 * np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
con = w0*np.random.uniform(size=(dim, dim))
np.fill_diagonal(con, 0)
lif = LIF_population(T=T, tau_m=20, tref=3,  weights=con, n_neurons=dim, Iext=input_step_function(t), verbose=0)
lif.run()

# Visualize
lif.plot_volt_trace(1)
lif.raster_plot()
lif.plot_populations(bins=1000, smoothing=True, sigma=15)


