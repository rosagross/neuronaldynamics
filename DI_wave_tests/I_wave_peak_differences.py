import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import matplotlib
from Model.DI_wave import DI_wave_simulation
from tqdm.contrib import itertools
from tqdm import tqdm
from Model.Neck import generate_EP
import scipy
from Utils import butter_highpass_filter, get_peak_values
matplotlib.use('TkAgg')

voltage_view = False
plot_E_fields = False
simulate = True

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]


E_values = np.linspace(150, 400, 100)
theta_values = np.linspace(0, 180, 100)
E_mesh, t_mesh = np.meshgrid(t ,E_values)
mesh_shapes = E_mesh.shape
z = np.zeros((len(theta_values), mesh_shapes[0], mesh_shapes[1]))

path_root = 'C:\\Users\\emueller'
# path_root = '/home/emueller'
fn_session = os.path.join(path_root, 'Downloads', 'gpc.pkl')
hdf5_path = os.path.join(path_root, '\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5')
save_file_path = os.path.join(path_root, '\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning')

simulation_name = 'E_theta_2D'
fname = os.path.join(save_file_path, simulation_name)
n_theta = theta_values.shape[0]
n_E = E_values.shape[0]
# out = np.zeros((E_values.shape[0], t.shape[0]))
# for i, E_i in tqdm(enumerate(E_values)):
#     parameters = {'intensity': E_i, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.5,
#                           'i_scale': 5.148136e-6, 'theta': 0, 'detrend': True,
#                           'fn_session': fn_session, 'T': T, 'name': 'E_theta_diw_single_run', 'dt': dt, 'min_delay': 0,
#                           'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
#                                                 'tau_ref': [0], #1.5
#                                                 'tau_mem': [12],
#                                                 'input_type': 'stochastic-current',
#                                                 'static_noise': True,
#                                                 'init_pdf_offset': 0,
#                                                 'init_pdf_sigma': 0.5,
#                                                 'init_pdf_weight': 0,
#                                                 'delay_kernel_type': 'alpha',
#                                                 'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3},
#                                                 'dv': dv,
#                                                 'dt': dt,
#                                                 'solver': 'Hu-2021',
#                                                 'current_sigma': 4,
#                                                 'verbose': 0,
#                                                 'tqdm_disable': True}}
#
#     di_model = DI_wave_simulation(parameters=parameters, logname=None)
#     di_model.simulate()
#     r_i = di_model.mass_model.r[0]
#     out[i] = r_i
    # peak_values = get_peak_values(t, r_i, plot=True)

# t_0_delta = np.zeros_like(E_values)
# amp_delta_peaks = np.zeros_like(E_values)
# for i in range(E_values.shape[0]):
#     peak_values_i = get_peak_values(t, out[i])
#     t_0_delta[i] = peak_values_i['t_delta_peaks'][0]
#     amp_delta_peaks[i] = peak_values_i['amp_delta_peaks'][0]
    #
# plt.plot(E_values, t_0_delta)
# plt.plot(E_values, amp_delta_peaks)

if simulate:
    for i, j in itertools.product(range(n_theta), range(n_E)):
            theta_i = theta_values[i]
            E_j = E_values[j]
            parameters = {'intensity': E_j, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.5,
                      'i_scale': 5.148136e-6, 'theta': theta_i, 'detrend': True,
                      'fn_session': fn_session, 'T': T, 'name': 'intensity_diw_single_run', 'dt': dt, 'min_delay': 0,
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
            z[i, j] = r_i
    with h5py.File(simulation_name + '.hdf5', 'w') as h5file:
        h5file.create_dataset('E_theta_2D', data=z)
    print(f'saved to {simulation_name}.hdf5')

if plot_E_fields:

    with h5py.File(simulation_name + '.hdf5', 'r') as h5file:
        data = np.array(h5file['E_theta_2D'])*1e3 # conversion from 1/ms to 1/s
    label = "r (Hz)"
    if voltage_view:
        label = ('V (a.u.)')
        for i, theta_i in enumerate(theta_values):
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

    fig, axs = plt.subplots(2, 4, figsize=(12, 9))
    z_max = data.max()
    for i, theta_i in enumerate(theta_values):
        row_idx = 0
        col_idx = i
        if i > 3:
            row_idx = 1
            col_idx -= 4

        ax = axs[row_idx, col_idx]

        z_i = data[i]
        pcm = ax.pcolormesh(E_mesh, t_mesh, z_i, shading="auto", cmap='gnuplot2', vmax=z_max)
        # ax.set_ylabel('t in (ms)')
        ax.set_title(f'phi: {theta_i}°')
        if col_idx == 0:
            ax.set_ylabel('|E| (V/m)')
        if row_idx > 0:
            ax.set_xlabel('t (ms)')
        ax.grid(True)
        ax.set_xticks([0, 2, 4, 5, 6, 7, 8, 10, 12])
        # cbar = fig.colorbar(pcm, ax=cbar_ax, label="Intensity", orientation="horizontal")
        # plt.tight_layout()
    # plt.tight_layout()

    # fig.subplots_adjust(right=0.7)
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.12, 0.8, 0.01])
    fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal", label=label)
    # plt.tight_layout()
    plt.show()

