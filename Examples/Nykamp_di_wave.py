import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import random
import pygpc
import os
import h5py
from tqdm import tqdm
from Model.Nykamp_Model import Nykamp_Model_1
from Model.Neck import generate_EP
from Utils import DI_wave_test_function, nrmse
matplotlib.use('TkAgg')

plot_convolution = False

# time in ms
t = np.linspace(0, 99.81, 500)
# t = np.linspace(0, 32., 1000)
dt = np.diff(t)[0]
T = t[-1] + dt

# scaling factor for current (gpc was done in normalized current space)
i_scale = 5.148136e-9

# read gpc session
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
fn_session = '/home/erik/Downloads/gpc.pkl'
session = pygpc.read_session(fname=fn_session)

with h5py.File(os.path.splitext(fn_session)[0] + ".hdf5", "r") as f:
    coeffs = f["coeffs"][:]

# create grid object to transform from real to normalized coordinates [-1, 1]
theta = 0               # angle of e-field [0, 180]°
gradient = 0            # relative gradient of e-field [-20, 20] %/mm
intensity = 160        # intensity of e-field [100, 400] V/m
fraction_nmda = 0.5     # fraction of nmda synapses [0.25, 0.75]
fraction_gaba_a = 0.95  # fraction of gaba_a synapses [0.9, 1.0]
fraction_ex = 0.7 # 0.40      # fraction of exc/ihn synapses [0.2, 0.8]

coords = np.array([[theta, gradient, intensity, fraction_nmda, fraction_gaba_a, fraction_ex]])

grid = pygpc.RandomGrid(parameters_random=session.parameters_random, coords=coords)

# use gpc approximation to compute current
current = session.gpc[0].get_approximation(coeffs, grid.coords_norm) * i_scale
current = current.flatten()

# set back half of current to 0
# current[300:] = 0

# convert to µA from A
ext_current = current * 1e6

# interpolate current on custom time grid
T_new = 20
dt_new = 0.01
t_new = np.arange(0, T_new, dt_new)
ext_current = np.interp(t_new, t, ext_current)


# # plot current
# plt.plot(t_new, ext_current)
# plt.xlabel('time in ms')
# plt.ylabel('Iext in A')
# plt.show()

y = DI_wave_test_function(t_new, intensity=2, t0=1.5, dt=1.5, width=0.3)

# plt.plot(t_new, y)
# plt.xlabel('time in ms')
# plt.ylabel('firing rate test function')
# plt.grid()
# plt.show()

# set a scalable conductance in mS?
# g_r_l5pt = 7e-5
# g_r_l5pt = 3.0e-5


dv = 0.1

pars_1D = {}
pars_1D['connectivity_matrix'] = np.array([[15]])
pars_1D['u_rest'] = -70
pars_1D['u_thr'] = -55
pars_1D['u_exc'] = 0
pars_1D['u_inh'] = -75
pars_1D['tau_mem'] = np.array([12])
# pars_1D['tau_ref'] = np.array([2.2])
pars_1D['tau_ref'] = np.array([1.0])
pars_1D['mu_gamma'] = np.array([[0.008, 0.027]])
pars_1D['var_coeff_gamma'] = 0.5*np.ones((1, 2))
# pars_1D['delay_kernel_parameters'] = {'tau_alpha': 1/3, 'n_alpha': 9}
pars_1D['delay_kernel_type'] = 'bi-exp'
pars_1D['delay_kernel_parameters'] = {'tau_1': 0.2, 'tau_2': 1.7, 'tau_cond': 1, 'g_peak': 1e-4}
# pars_1D['synapse_pdf_type'] = 'log-normal'
# mu_vals = np.array([0.008, 0.008])
# sigma_vals = np.array([0.004, 0.004])
# pars_1D['synapse_pdf_params'] = np.array([[mu_vals], [sigma_vals]])
pars_1D['input_function_type'] = 'custom'
pars_1D['input_function_idx'] = [0, 0]
pars_1D['population_type'] = ['exc']
pars_1D['input_type'] = 'current'
pars_1D['input_function'] = ext_current # * 0.33 # scaling down by 3
# pars_1D['g_leak'] = [g_r_l5pt]
pars_1D['T'] = T_new
pars_1D['dt'] = dt_new
pars_1D['dv'] = dv
pars_1D['sparse_mat'] = True
pars_1D['name'] = 'Nykamp'
nyk1D = Nykamp_Model_1(parameters=pars_1D, name='Nykamp')

# nyk1D.plot_delay_kernel()

nyk1D.simulate()
nyk1D.plot(heat_map=True, plot_input=True)
nyk1D.save_log()
nyk1D.clean()

nykamp_rate = nyk1D.r[0]

EP, t_EP, AP_out = generate_EP(d=0.1, plot=False, Axontype=1, dt=dt_new*10)
EP = -EP
EP = EP / np.max(EP)
EP_small = np.interp(t_new[t_new < 1.0] - 0.5, t_EP, EP)
nykamp_potential = scipy.signal.convolve(nykamp_rate, EP_small)
nykamp_shape = nykamp_rate.shape[0]
nykamp_potential_out = nykamp_potential[:nykamp_shape]
di_max = np.max(y)
nykamp_potential_scaled = nykamp_potential_out / np.max(nykamp_potential_out) * di_max
if plot_convolution:
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t_new, nykamp_rate)
    ax[0].set_ylabel('DI wave potential')
    ax[1].plot(t_new[:EP_small.shape[0]], EP_small)
    ax[1].set_ylabel('Kernel')
    ax[2].plot(t_new, nykamp_potential_scaled)
    ax[2].set_ylabel('DI wave rate')
    for i in range(3):
        ax[i].set_xlabel('t (ms)')
        ax[i].set_xlim([t_new[0], t_new[-1]])
    plt.show()


diff = nrmse(y, nykamp_potential_scaled)
# plt.plot(t_new, nykamp_rate)
plt.plot(t_new, nykamp_potential_scaled)
plt.plot(t_new, y)
plt.grid()
plt.xlabel('t in ms')
plt.legend(['nykamp_potential', 'D-I-wave test function'])
# plt.legend(['nykamp rate', 'nykamp_potential', 'D-I-wave test function'])
plt.title(f'nrmse: {diff:.4f}')
plt.show()


