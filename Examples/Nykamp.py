"""
Python implementation of Nykamp et al. 2000 population density model for 3 populations.
Authors and editors: Konstantin Weise, Aaron Miller, Erik Müller
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
import os
from Model.Nykamp_Model import Nykamp_Model, Nykamp_Model_1
matplotlib.use('TkAgg')



# init parameters
def input_sine_function(t):
    v0 = 0.7
    f = 10
    return v0 * (1 + np.sin(2*np.pi*f*t/1000))

### single population test ###
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
pars_1D['input_function'] = input_sine_function
pars_1D['input_function_type'] = 'custom'
pars_1D['input_function_idx'] = [0, 0]
pars_1D['population_type'] = ['exc']

T = 100 # 200
dt = 0.1 # 0.1
dv = 0.01

nyk1D = Nykamp_Model_1(parameters=pars_1D, name='nykamp_test_1D')
nyk1D.simulate(T=T, dt=dt, dv=dv, verbose=0, sparse_mat=True)
nyk1D.plot(heat_map=True)
nyk1D.clean()

parameters = {}
w0 = 30
# TODO: if this factor is too high, rates and voltage pdf becomes inf! Find out why
parameters['connectivity_matrix'] = w0*np.array([[1/2, 1], [1, 1]])
parameters['u_rest'] = -65
parameters['u_thr'] = -55
parameters['u_exc'] = 0
parameters['u_inh'] = -70
parameters['tau_mem'] = np.array([20, 10])
parameters['tau_ref'] = np.array([3, 1])
parameters['mu_gamma'] = np.array([[0.008, 0.027], [0.020, 0.066]])
parameters['var_coeff_gamma'] = 0.5*np.ones((2, 2))
parameters['tau_alpha'] = 1/3
parameters['n_alpha'] = 9
parameters['input_function'] = input_sine_function
parameters['input_function_type'] = 'custom'
parameters['input_function_idx'] = [0, 0]
parameters['population_type'] = ['exc', 'inh']

T = 100 # 200
dt = 0.1 # 0.1
dv = 0.01

nyk = Nykamp_Model_1(parameters=parameters, name='nykamp_test_2D')
nyk.simulate(T=T, dt=dt, dv=dv, verbose=0, sparse_mat=True)

parameters_1 = parameters.copy()
parameters_1['connectivity_matrix'] = np.array([[15, 30, 30], [30, 30, 30], [30, 30, 30]])
parameters_1['tau_mem'] = np.array([20, 10, 20])
parameters_1['tau_ref'] = np.array([3, 1, 3])
parameters_1['mu_gamma'] = np.array([[0.008, 0.027], [0.020, 0.066], [0.008, 0.027]])
parameters_1['var_coeff_gamma'] = 0.5*np.ones((3, 2))
parameters_1['population_type'] = ['exc', 'inh', 'exc']
parameters_1['input_function_idx'] = [0, 0]
nyk_1 = Nykamp_Model_1(parameters=parameters_1, name='nykamp_test_3D')
# nyk_1.simulate(T=T, dt=dt, dv=dv)

# plot results
nyk.plot('nykamp_test_2D', heat_map=True)
nyk.clean()
# plot('nykamp_test_3D', heat_map=True)



#############################################################################################

# test sparse vs non-spares
dts = np.array([0.1, 0.2, 0.5, 1, 2])
dvs = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
T = 50
n_repetitions = 10


import time
def model_timing(n_repetitions, dts, dvs, T, model):

    ts_t = np.zeros((dts.shape[0], n_repetitions))
    ts_t_sparse = np.zeros_like(ts_t)
    ts_v = np.zeros_like(ts_t)
    ts_v_sparse = np.zeros_like(ts_t)

    for n in range(n_repetitions):
        print(f'>n = {n + 1} ---------------------------------------')
        for i in range(5):
            t0 = time.time()
            model.simulate(T=T, dt=dts[i], dv=0.01, sparse_mat=False)
            t1 = time.time()

            t0_sparse = time.time()
            model.simulate(T=T, dt=dts[i], dv=0.01, sparse_mat=True)
            t1_sparse = time.time()

            t_n = t1-t0
            t_n_sparse = t1_sparse - t0_sparse

            ts_t[i, n] = t_n
            ts_t_sparse[i, n] = t_n_sparse

            if i != 0:
                t0 = time.time()
                model.simulate(T=T, dt=0.1, dv=dvs[i], sparse_mat=False)
                t1 = time.time()

                t0_sparse = time.time()
                model.simulate(T=T, dt=0.1, dv=dvs[i], sparse_mat=True)
                t1_sparse = time.time()

            t_n = t1 - t0
            t_n_sparse = t1_sparse - t0_sparse

            ts_v[i, n] = t_n
            ts_v_sparse[i, n] = t_n_sparse

            with h5py.File('speed_test.hdf5', 'w') as h5file:
                h5file.create_dataset('dts', data=dts)
                h5file.create_dataset('dvs', data=dvs)
                h5file.create_dataset('T', data=T)
                h5file.create_dataset('n_repetitions', data=n_repetitions)
                h5file.create_dataset('ts_t', data=ts_t)
                h5file.create_dataset('ts_t_sparse', data=ts_t_sparse)
                h5file.create_dataset('ts_v', data=ts_v)
                h5file.create_dataset('ts_v_sparse', data=ts_v_sparse)
def plot_timing(fname):
    with h5py.File(fname + '.hdf5', 'r') as h5file:
        dts = np.array(h5file['dts'])
        dvs = np.array(h5file['dvs'])
        T = np.array(h5file['T'])
        n_repetitions = np.array(h5file['n_repetitions'])
        ts_t = np.array(h5file['ts_t'])
        ts_t_sparse = np.array(h5file['ts_t_sparse'])
        ts_v = np.array(h5file['ts_v'])
        ts_v_sparse = np.array(h5file['ts_v_sparse'])

    ##########################################################################
    # ax # 1
    ##########################################################################

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)

    mean_t = np.mean(ts_t, axis=1)[::-1]
    t_vals = dts[::-1]
    xticks = [k for k in t_vals]
    t_min, t_max = np.min(ts_t, axis=1), np.max(ts_t, axis=1)
    t_errors = t_max - t_min

    mean_t_s = np.mean(ts_t_sparse, axis=1)[::-1]
    t_min_s, t_max_s = np.min(ts_t_sparse, axis=1), np.max(ts_t_sparse, axis=1)
    t_errors_s = t_max_s - t_min_s

    ax.errorbar(t_vals, mean_t, yerr=t_errors, fmt='x', c='red')
    ax.plot(t_vals, mean_t, c='red', alpha=0.4, linestyle='--')

    ax.errorbar(t_vals, mean_t_s, yerr=t_errors_s, fmt='o', c='blue')
    ax.plot(t_vals, mean_t_s, c='blue', alpha=0.4, linestyle='--')

    ax.set_title(f"Computation time over dt for T=50ms simulation")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(['original', 'sparse'])
    ax.set_ylim(1e0, 1e3)
    ax.set_ylabel("t in s")
    ax.set_xticks(xticks)
    ax.set_xlabel("dt")
    ax.grid()

    ##########################################################################
    # ax # 2
    ##########################################################################

    ax = fig.add_subplot(1, 2, 2)
    mean_t = np.mean(ts_v, axis=1)[::-1]
    v_vals = dvs[::-1]
    xticks = [k for k in v_vals]
    t_min, t_max = np.min(ts_v, axis=1), np.max(ts_v, axis=1)
    t_errors = t_max - t_min

    mean_t_s = np.mean(ts_v_sparse, axis=1)[::-1]
    t_min_s, t_max_s = np.min(ts_v_sparse, axis=1), np.max(ts_v_sparse, axis=1)
    t_errors_s = t_max_s - t_min_s

    ax.errorbar(v_vals, mean_t, yerr=t_errors, fmt='x', c='red')
    ax.plot(v_vals, mean_t, c='red', alpha=0.4, linestyle='--')

    ax.errorbar(v_vals, mean_t_s, yerr=t_errors_s, fmt='o', c='blue')
    ax.plot(v_vals, mean_t_s, c='blue', alpha=0.4, linestyle='--')

    ax.set_title(f"Computation time over dv for T=50ms simulation")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(['original', 'sparse'])
    ax.set_ylim(1e0, 1e3)
    ax.set_ylabel("t in s")
    ax.set_xticks(xticks)
    ax.set_xlabel("dv")

    ax.grid()
    plt.tight_layout()
    plt.show()



# model_timing(n_repetitions=n_repetitions, dts=dts, dvs=dvs, T=T, model=nyk)
# plot_timing('speed_test')
