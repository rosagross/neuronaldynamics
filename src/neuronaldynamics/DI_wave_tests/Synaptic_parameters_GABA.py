import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import matplotlib
from neuronaldynamics.Model.DI_wave import DI_wave_simulation
from tqdm.contrib import itertools
from neuronaldynamics.Model import generate_EP
import scipy
from Utils import butter_highpass_filter, delay_signal

matplotlib.use('TkAgg')

voltage_view = True
plot_params = True
single_shot_plot = False
simulate = True
delay = True

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]

E_values = [175, 225, 250, 300, 350, 400]
p_values = np.linspace(0.9, 1.0, 100)
n_E = len(E_values)
n_P = p_values.shape[0]

# create mesh
p_mesh, t_mesh = np.meshgrid(t, p_values)
mesh_shapes = p_mesh.shape
z = np.zeros((n_E, mesh_shapes[0], mesh_shapes[1]))

# path_root = '/home/erik/Downloads/'
path_root = 'C:\\Users\\emueller'
# path_root = 'C:\\Users\\User'

# fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
fn_session = os.path.join(path_root, 'Downloads', 'gpc.pkl')
hdf5_path = os.path.join(path_root, '\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5')
save_file_path = os.path.join(path_root, '\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning')

# hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
# hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"

# save_file_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning"
# save_file_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning"
simulation_name = 'GABAa_GABAb_diw'
fname = os.path.join(save_file_path, simulation_name)

# simulate
if simulate:
    for i, j in itertools.product(range(n_P), range(n_E)):
        p_val_i = p_values[i]
        E_val_j = E_values[j]
    # for i in tqdm(range(p_values.shape[0])):
        parameters = {'intensity': E_val_j, 'fraction_nmda': 0.61, 'fraction_gaba_a': p_val_i, 'fraction_ex': 0.5,
                  'i_scale': 5.148136e-6, 'theta': 0, 'detrend': True, 'delay_signal': True,
                  'fn_session': fn_session, 'T': T, 'name': 'param_diw_single_run_2', 'dt': dt, 'min_delay': 0,
                  'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
                                        'tau_ref': [0], #1.5
                                        'tau_mem': [12],
                                        'input_type': 'stochastic-current',
                                        'static_noise': True,
                                        'init_pdf_offset': 0,
                                        'init_pdf_sigma': 0.5,
                                        'init_pdf_weight': 0,
                                        'delay_kernel_type': 'alpha',
                                        'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3},
                                        'dv': dv,
                                        'dt': dt,
                                        'solver': 'Hu-2021',
                                        'current_sigma': 4,
                                        'verbose': 0,
                                        'tqdm_disable': True}}

        di_model = DI_wave_simulation(parameters=parameters, logname=None)
        di_model.simulate()
        r_i = di_model.mass_model.r[0]
        z[j, i] = r_i
    with h5py.File(simulation_name + '.hdf5', 'w') as h5file:
        h5file.create_dataset('syn_parameter_data', data=z)
    print(f'saved to {simulation_name}.hdf5')

if plot_params:

    with h5py.File(simulation_name + '.hdf5', 'r') as h5file:
        data = np.array(h5file['syn_parameter_data'])*1e3 # conversion from 1/ms to 1/s
    label = "r (Hz)"
    if voltage_view:
        label = ('V (a.u.)')
        for i, E_i in enumerate(E_values):
            EP, t_EP, AP_out = generate_EP(d=0.1, plot=False, Axontype=1, dt=dt * 10)
            EP = -EP
            EP = EP / np.max(EP)
            EP_small = np.interp(t[t < 1.0] - 0.5, t_EP, EP)
            for j in range(data[i].shape[0]):
                nmm_potential = scipy.signal.convolve(data[i, j], EP_small)
                nmm_shape = data[i, j].shape[0]
                nmm_potential_out = nmm_potential[:nmm_shape]

                v_out_hp = butter_highpass_filter(nmm_potential_out, cutoff=0.05,
                                                  fps=int(1 / dt))  # very small cutoff
                v_out_mean = nmm_potential_out.mean()
                data[i, j] = v_out_hp/1000

    if delay:
        for i in range(n_E):
            for j in range(n_P):
                data[i, j] = delay_signal(data[i, j], 2, dt=dt)

    fig, axs = plt.subplots(2, 3, figsize=(12, 9))
    z_max = data.max()
    for i, E_i in enumerate(E_values):
        row_idx = 0
        col_idx = i
        if i > 2:
            row_idx = 1
            col_idx -= 3

        ax = axs[row_idx, col_idx]

        z_i = data[i]
        pcm = ax.pcolormesh(p_mesh, t_mesh, z_i, shading="auto", cmap='gnuplot2', vmax=z_max)
        # ax.set_ylabel('t in (ms)')
        ax.set_title(f'|E|: {E_i} V/m')
        if col_idx == 0:
            ax.set_ylabel('GABAa/GABAb ratio')
        if row_idx > 0:
            ax.set_xlabel('t (ms)')
        ax.grid(True)
        ax.set_xticks([0, 2, 4, 5, 6, 7, 8, 10, 12])
        # cbar = fig.colorbar(pcm, ax=cbar_ax, label="Intensity", orientation="horizontal")
        # plt.tight_layout()
    # plt.tight_layout()

    # fig.subplots_adjust(right=0.7)
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.12, 0.12, 0.8, 0.01])
    fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal", label=label)
    # plt.tight_layout()
    plt.show()




# plot single shot
if single_shot_plot:
    with h5py.File(simulation_name + '.hdf5', 'r') as h5file:
        data = np.array(h5file['syn_parameter_data'])*1e3 # conversion from 1/ms to 1/s
    z = data
    label = ('r (Hz)')
    if voltage_view:
        label = ('V (a.u.)')
        EP, t_EP, AP_out = generate_EP(d=0.1, plot=False, Axontype=1, dt=dt * 10)
        EP = -EP
        EP = EP / np.max(EP)
        EP_small = np.interp(t[t < 1.0] - 0.5, t_EP, EP)
        for j in range(z.shape[0]):
            nmm_potential = scipy.signal.convolve(z[j], EP_small)
            nmm_shape = z[j].shape[0]
            nmm_potential_out = nmm_potential[:nmm_shape]

            v_out_hp = butter_highpass_filter(nmm_potential_out, cutoff=0.05,
                                              fps=int(1 / dt))  # very small cutoff
            v_out_mean = nmm_potential_out.mean()
            z[j] = v_out_hp / 1000
    z_max = z.max()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))


    pcm = ax.pcolormesh(p_mesh, t_mesh, z, shading="auto", cmap='gnuplot2', vmax=z_max)
    ax.set_title(f'Variation of GABAa/GABAb ratio')
    ax.set_ylabel('GABAa/GABAb ratio')
    ax.set_xlabel('t (ms)')
    ax.grid(True)
    ax.set_xticks([0, 2, 4, 5, 6, 7, 8, 10, 12])
    plt.tight_layout()
    fig.colorbar(pcm, label=label)
    plt.show()