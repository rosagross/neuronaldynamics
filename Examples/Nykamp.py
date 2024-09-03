"""
Pythonic implementation of Nykamp et al. 2000 population density model for 3 populations.
Authors and editors: Konstantin Weise, Aaron Miller, Erik Müller
"""
import numpy as np
import matplotlib.pyplot as plt
from Model.Nykamp import Nykamp_Model
import h5py
import matplotlib
import os
from Model.Nykamp import Nykamp_Model_1
matplotlib.use('TkAgg')

def plot(fname, heat_map=False, plot_idxs=None):
    with h5py.File(fname + '.hdf5', 'r') as h5file:

        t_plot = np.array(h5file['t'])
        v = np.array(h5file['v'])
        r_plot = np.array(h5file['r'])
        rho_plot = np.array(h5file['rho_plot'])
        p_types_raw = h5file['p_types']
        p_types = p_types_raw.asstr()[:]

    if plot_idxs is None:
        n_plots = len(p_types)
        plot_idxs = np.arange(n_plots)
    else:
        n_plots = len(plot_idxs)

    fig = plt.figure(figsize=(10, 8.5))
    for i_plot, plot_idx in enumerate(plot_idxs):
        plot_loc_1 = int(2*i_plot + 1)
        plot_loc_2 = int(2 * i_plot + 2)
        if heat_map:
            ax = fig.add_subplot(n_plots, 2, plot_loc_1)
            X, Y = np.meshgrid(t_plot, v)
            z_min, z_max = 0, np.abs(rho_plot[plot_idx]).max()
            c = ax.pcolormesh(X, Y, rho_plot[plot_idx], cmap='viridis', vmin=z_min, vmax=z_max)
            fig.colorbar(c, ax=ax)

        else:
            ax = fig.add_subplot(n_plots, 2, plot_loc_1, projection='3d')
            X, Y = np.meshgrid(t_plot, v)
            ax.plot_surface(X, Y, rho_plot[plot_idx],
                            cmap="jet", linewidth=0, antialiased=False, rcount=100, ccount=100)
            ax.set_zlim3d(0, 1)

        ax.set_title(f"Membrane potential distribution ({str(p_types[plot_idx])})")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("membrane potential (mv)")

        ax = fig.add_subplot(n_plots, 2, plot_loc_2)
        ax.plot(t_plot, r_plot[plot_idx] * 1000)
        ax.set_title(f"Population activity ({str(p_types[plot_idx])})")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("time (ms)")
        ax.grid()
    plt.tight_layout()
    plt.show()

# init parameters
def input_sine_function(t):
    v0 = .7
    f = 10
    return v0 * (1 + np.sin(2*np.pi*f*t/1000))

parameters = {}
parameters['connectivity_matrix'] = np.array([[15, 30], [30, 30]])
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

T = 200 # 200
dt = 0.1 # 0.1
dv = 0.01

nyk = Nykamp_Model_1(parameters=parameters, name='nykamp_test_2D')
# nyk.simulate(T=T, dt=dt, dv=dv)

parameters_1 = parameters.copy()
parameters_1['connectivity_matrix'] = np.array([[15, 30, 30], [30, 30, 30], [30, 30, 30]])
parameters_1['tau_mem'] = np.array([20, 10, 20])
parameters_1['tau_ref'] = np.array([3, 1, 3])
parameters_1['mu_gamma'] = np.array([[0.008, 0.027], [0.020, 0.066], [0.008, 0.027]])
parameters_1['var_coeff_gamma'] = 0.5*np.ones((3, 2))
parameters_1['population_type'] = ['exc', 'inh', 'exc']
nyk_1 = Nykamp_Model_1(parameters=parameters_1, name='nykamp_test_3D')
# nyk_1.simulate(T=T, dt=dt, dv=dv)

# plot results
plot('nykamp_test_2D', heat_map=True)
plot('nykamp_test_3D', heat_map=True)

# os.remove('nykamp_test_2D' + '.hdf5')
# os.remove('nykamp_test_3D' + '.hdf5')