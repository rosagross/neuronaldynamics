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

# time in ms
t = np.linspace(0, 99.81, 500)
# t = np.linspace(0, 32., 1000)
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
fraction_ex = 0.6      # fraction of exc/ihn synapses [0.2, 0.8]

coords = np.array([[theta, gradient, intensity, fraction_nmda, fraction_gaba_a, fraction_ex]])

grid = pygpc.RandomGrid(parameters_random=session.parameters_random, coords=coords)

# use gpc approximation to compute current
current = session.gpc[0].get_approximation(coeffs, grid.coords_norm) * i_scale
current = current.flatten()

# # set back half of current to 0
current[300:] = 0

# convert to µA from A
ext_current = current * 1e6
# plot current
plt.plot(t, current)
plt.xlabel('time in ms')
plt.ylabel('Iext in A')
plt.show()


# set a scalable conductance in mS?
g_r_l5pt = 7e-5

dv = 0.1

pars_1D = {}
pars_1D['connectivity_matrix'] = np.array([[10]])
pars_1D['u_rest'] = -70
pars_1D['u_thr'] = -55
pars_1D['u_exc'] = 0
pars_1D['u_inh'] = -75
pars_1D['tau_mem'] = np.array([200])
pars_1D['tau_ref'] = np.array([3])
pars_1D['mu_gamma'] = np.array([[0.008, 0.027]])
pars_1D['var_coeff_gamma'] = 0.5*np.ones((1, 2))
pars_1D['tau_alpha'] = 1/3
pars_1D['n_alpha'] = 9
pars_1D['input_function_type'] = 'custom'
pars_1D['input_function_idx'] = [0, 0]
pars_1D['population_type'] = ['exc']
pars_1D['input_type'] = 'current'
pars_1D['input_function'] = ext_current * 3e-1 # scaling down by 20
pars_1D['g_leak'] = [g_r_l5pt]
nyk1D = Nykamp_Model_1(parameters=pars_1D, name='Nykamp')
nyk1D.simulate(T=T, dt=dt, dv=dv, verbose=0, sparse_mat=True)
nyk1D.plot(heat_map=True)
nyk1D.save_log()
nyk1D.clean()


