import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import random
import pygpc
import os
import h5py
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
con = np.zeros((dim, dim))
n_connection = 220  # int(dim/10)

for i in tqdm(range(dim), f'computing random neuron connections for {dim} neurons'):
    possible_connections = np.arange(dim)
    # exclude current idx
    possible_connections = possible_connections[possible_connections!=i]
    connections_i = random.sample(possible_connections.tolist(), n_connection)
    if np.unique(np.array(connections_i)).shape[0] != n_connection:
        print(f"failed sampling!!!, idx {i}")
    con[i, connections_i] = 1

def i_ext(t):
    f = 10
    i_ext_0 = 0.7e2  # 200µA / 10 mS  =  20mV input
    return i_ext_0 * (1 + np.sin(2*np.pi*f/1000*t))

def i_ext_1(t):
    f = 10
    i_ext_0 = 0.7*5e0 # 200µA / 10mS  =  20mV input
    return i_ext_0 * (1 + np.sin(2*np.pi*f/1000*(t)))

def i_ext_population(t):
    f = 10
    i_ext_0 = 0.7e-2 * dim * (1/lif.g_r) * 1.5
    x0 = i_ext_0 / dim * lif.g_r * 100
    t0 = np.exp(-(x0 - 3)) + 2
    return i_ext_0 * (1 + np.sin(2*np.pi*f/1000*(t-t0)))

def step(t):
    t1 = 20
    t2 = 90
    i_0 = 1e-2
    res = np.zeros_like(t)
    res[t > t1] = i_0
    res[t > t2] = 0
    return res

def step_population(t):
    t1 = 20
    t2 = 90
    i_0 = 1e-2 * dim * (1/lif.g_r) * 1.5
    x0 = i_0 / dim * lif.g_r * 100
    t0 = np.exp(-(x0 - 3)) + 2
    res = np.zeros_like(t)
    res[t-t0 > t1] = i_0
    res[t-t0 > t2] = 0
    return res

i_ext_vals = i_ext(t)
# i_ext_vals = step(t)
i_ext_vals = i_ext_vals.repeat(dim).reshape(t.shape[0], dim).T

neuron_parameters = {'T': T, 'tau_m': 20, 't_ref': 3, 'weights': con, 'n_neurons': dim, 'Iinj': i_ext_vals}
lif = Conductance_LIF(parameters=neuron_parameters)
lif.run()

# visualize
# lif.plot_volt_trace(idx=3, population_idx=2)
lif.raster_plot()
lif.plot_populations(bins=1000, smoothing=True, sigma=10, hide_refractory=True, cutoff=None, size=1)

########################################################################################################################
# Nykamp model #
########################################################################################################################

def no_in(t):
    return 0

pars_1D = {}
pars_1D['connectivity_matrix'] = 30*np.array([[1/2]])
pars_1D['u_rest'] = -65
pars_1D['u_thr'] = -55
pars_1D['u_exc'] = 0
pars_1D['u_inh'] = -70
pars_1D['tau_mem'] = np.array([20])
pars_1D['tau_ref'] = np.array([3])
pars_1D['mu_gamma'] = np.array([[0.008, 0.027]])
pars_1D['var_coeff_gamma'] = 0.5*np.ones((1, 2))
pars_1D['tau_alpha'] = 1/3
pars_1D['n_alpha'] = 9
# pars_1D['input_function'] = step_population
pars_1D['input_function'] = i_ext_1
pars_1D['input_function_type'] = 'custom'
pars_1D['input_function_idx'] = [0, 0]
pars_1D['population_type'] = ['exc']

g_leak = lif.g_r
pars_1D['input_type'] = 'current'
pars_1D['g_leak'] = [g_leak]  # leakage conductance

dt = 0.1 # 0.1
dv = 0.01

nyk1D = Nykamp_Model_1(parameters=pars_1D, name='Nykamp')
nyk1D.simulate(T=T, dt=dt, dv=dv, verbose=0, sparse_mat=True)
nyk1D.plot(heat_map=True)
#
compare_firing_rate('Nykamp', 'Conductance_LIF')

########################################################################################################################
# gpc input current
########################################################################################################################

# time in ms
t = np.linspace(0, 99.81, 500)
dt = np.diff(t)[0]
T = t[-1] + dt

# scaling factor for current (gpc was done in normalized current space)
i_scale = 5.148136e-9

# read gpc session
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
session = pygpc.read_session(fname=fn_session)

with h5py.File(os.path.splitext(fn_session)[0] + ".hdf5", "r") as f:
    coeffs = f["coeffs"][:]

# create grid object to transform from real to normalized coordinates [-1, 1]
theta = 0               # angle of e-field [0, 180]°
gradient = 0            # relative gradient of e-field [-20, 20] %/mm
intensity = 250         # intensity of e-field [100, 400] V/m
fraction_nmda = 0.5     # fraction of nmda synapses [0.25, 0.75]
fraction_gaba_a = 0.95  # fraction of nmda synapses [0.9, 1.0]
fraction_ex = 0.5       # fraction of exc/ihn synapses [0.2, 0.8]

coords = np.array([[theta, gradient, intensity, fraction_nmda, fraction_gaba_a, fraction_ex]])

grid = pygpc.RandomGrid(parameters_random=session.parameters_random, coords=coords)

# use gpc approximation to compute current
current = session.gpc[0].get_approximation(coeffs, grid.coords_norm) * i_scale
current = current.flatten()

# convert to µA from A
ext_current = current * 1e6
# plot current
plt.plot(t, current)
plt.xlabel('time in ms')
plt.ylabel('Iext in A')
plt.show()


# set a scalable conductance in mS?
g_r_l5pt = 5e-5
ext_curr_vals = ext_current.repeat(dim).reshape(t.shape[0], dim).T

con = np.zeros((dim, dim))
n_connection = 200  # int(dim/10)

for i in tqdm(range(dim), f'computing random neuron connections for {dim} neurons'):
    possible_connections = np.arange(dim)
    # exclude current idx
    possible_connections = possible_connections[possible_connections!=i]
    connections_i = random.sample(possible_connections.tolist(), n_connection)
    if np.unique(np.array(connections_i)).shape[0] != n_connection:
        print(f"failed sampling!!!, idx {i}")
    con[i, connections_i] = 1

neuron_parameters = {'T': T, 'dt': dt, 'tau_m': 20, 't_ref': 3, 'weights': con, 'n_neurons': dim, 'Iinj': ext_curr_vals,
                     'g_r': g_r_l5pt}


lif = Conductance_LIF(parameters=neuron_parameters)
lif.run()
lif.raster_plot()
lif.plot_populations(bins=1000, smoothing=True, sigma=10, hide_refractory=True, cutoff=None, size=1)

pars_1D['input_function'] = ext_current * 5e-2 # scaling down by 20
pars_1D['g_leak'] = [g_r_l5pt]
nyk1D = Nykamp_Model_1(parameters=pars_1D, name='Nykamp')
nyk1D.simulate(T=T, dt=dt, dv=dv, verbose=0, sparse_mat=True)
nyk1D.plot(heat_map=True)
compare_firing_rate('Nykamp', 'Conductance_LIF')

