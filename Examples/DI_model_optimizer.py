import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import scipy
import random
import pygpc
import os
import h5py
from tqdm import tqdm
from Model.LIF import Conductance_LIF
from Model.Nykamp_Model import Nykamp_Model_1
from Utils import compare_firing_rate, DI_wave_test_function, nrmse
matplotlib.use('TkAgg')

# time in ms
t = np.linspace(0, 99.81, 500)
# t = np.linspace(0, 32., 1000)
dt = np.diff(t)[0]
T = t[-1] + dt

g_r_l5pt = 7e-5

dv = 0.1

pars_1D = {}
pars_1D['connectivity_matrix'] = np.array([[20]])
pars_1D['u_rest'] = -70
pars_1D['u_thr'] = -55
pars_1D['u_exc'] = 0
pars_1D['u_inh'] = -75
pars_1D['tau_mem'] = np.array([20])
pars_1D['tau_ref'] = np.array([2.2])
pars_1D['mu_gamma'] = np.array([[0.008, 0.027]])
pars_1D['var_coeff_gamma'] = 0.5*np.ones((1, 2))
# pars_1D['delay_kernel_parameters'] = {'tau_alpha': 1/3, 'n_alpha': 9}
pars_1D['delay_kernel_type'] = 'bi-exp'
pars_1D['delay_kernel_parameters'] = {'tau_1': 0.2, 'tau_2': 1.7, 'tau_cond': 1, 'g_peak': 1e-4}
pars_1D['input_function_type'] = 'custom'
pars_1D['input_function_idx'] = [0, 0]
pars_1D['population_type'] = ['exc']
pars_1D['input_type'] = 'current'
pars_1D['tqdm_disable'] = True

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
fraction_gaba_a = 0.95  # fraction of gaba_a synapses [0.9, 1.0]
fraction_ex = 0.6      # fraction of exc/ihn synapses [0.2, 0.8]

y = DI_wave_test_function(t, intensity=1.5, t0=1, dt=1.5, width=0.3)
def simulate(intensity, fraction_nmda, fraction_gaba_a, fraction_ex, g_r_l5pt, idx='0'):
    coords = np.array([[theta, gradient, intensity, fraction_nmda, fraction_gaba_a, fraction_ex]])

    grid = pygpc.RandomGrid(parameters_random=session.parameters_random, coords=coords)

    # use gpc approximation to compute current
    current = session.gpc[0].get_approximation(coeffs, grid.coords_norm) * i_scale
    current = current.flatten()

    # # set back half of current to 0
    current[300:] = 0

    # convert to µA from A
    ext_current = current * 1e6

    pars_1D['input_function'] = ext_current # * 0.33 # scaling down by 3
    pars_1D['g_leak'] = [g_r_l5pt]


    # set a scalable conductance in mS?

    nyk1D = Nykamp_Model_1(parameters=pars_1D, name='Nykamp_' + idx)

    nyk1D.simulate(T=T, dt=dt, dv=dv, verbose=0, sparse_mat=True)
    nyk1D.plot(savefig=True, heat_map=True)
    nyk1D.clean()
    return nyk1D.r[0]

lower_bound = np.array([100, 0.2, 1e-5])
upper_bound = np.array([400, 0.8, 1e-4])

i = 0
min_error = 1.
max_iter = 10
n_param = lower_bound.shape[0]
eps = 0.01
errors = []
param_list = []

for i in tqdm(range(max_iter)):
    n_grid = 20
    param_values = np.zeros((n_param, n_grid))
    for j in range(n_param):
        param_values[j] = np.random.uniform(lower_bound[j], upper_bound[j], n_grid)
    error = np.zeros(param_values.shape[1])
    for k in range(n_grid):
        x = simulate(intensity=param_values[0, k], fraction_nmda=0.5, fraction_gaba_a=0.95,
                     fraction_ex=param_values[1, k],
                     g_r_l5pt=param_values[2, k], idx=f'{i}_{k}')
        error[k] = nrmse(y, x)
    min_error = np.nanmin(error)
    print(f'error: {min_error:.5f}')
    min_error_idx = np.nanargmin(error)
    errors.append(min_error)
    param_list.append(param_values[:, min_error_idx])
    print(param_values[:, min_error_idx])
    if min_error < eps:
        print(f'error: {min_error}')
        break
    # get new bounds for next iteration
    p_new = param_values[:, min_error_idx]
    delta = upper_bound - lower_bound
    for j in range(n_param):
        lower_bound[j] = max(lower_bound[j], p_new[j] - 0.5 * delta[j])
        upper_bound[j] = min(upper_bound[j], p_new[j] + 0.5 * delta[j])

plt.close()
nykamp_rate = x
diff = nrmse(y, nykamp_rate)
plt.plot(t, nykamp_rate)
plt.plot(t, y)
plt.grid()
plt.xlabel('t in ms')
plt.legend(['nykamp', 'D-I-wave test function'])
plt.title(f'nrmse: {diff:.4f}')
plt.show()


