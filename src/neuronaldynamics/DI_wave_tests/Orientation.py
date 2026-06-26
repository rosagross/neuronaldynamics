import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import matplotlib
from Model.DI_wave import DI_wave_simulation
from tqdm.contrib import itertools
from tqdm import tqdm
from Model import generate_EP
import scipy
from Utils import butter_highpass_filter, delay_signal
matplotlib.use('TkAgg')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

voltage_view = True
plot_orientations = True
simulate = False
single_shot = False

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]


thetas = np.linspace(0, np.pi, 100)
theta_deg = thetas / (2 * np.pi) * 360
r, theta_grid = np.meshgrid(t, thetas)
E_values = [175, 225, 250, 300, 350, 400]
single_shot_E = 200
r_shapes = r.shape
z = np.zeros((len(E_values), r_shapes[0], r_shapes[1]))

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'

# hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"

# save_file_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning"
save_file_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning"

simulation_name = 'diw_orientation_test'
fname = os.path.join(save_file_path, simulation_name)
n_theta = theta_deg.shape[0]
n_E = len(E_values)

plot_thetas = [k*45 for k in range(8)]
# plot_theta_idxs = []
# for plot_theta in plot_thetas:
#     theta_idx = np.where(theta_deg > plot_theta)[0][0]
#     plot_theta_idxs.append(theta_idx)

if simulate:
    for i, j in itertools.product(range(n_theta), range(n_E)):
            theta_i = theta_deg[i]
            E_j = E_values[j]
            parameters = {'intensity': E_j, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.5,
                      'i_scale': 5.148136e-6, 'theta': theta_i, 'detrend': True,
                      'fn_session': fn_session, 'T': T, 'name': 'orientation_diw_single_run', 'dt': dt, 'min_delay': 0,
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
                                            'voltage_sigma': 4,
                                            'verbose': 0,
                                            'tqdm_disable': True}}

            di_model = DI_wave_simulation(parameters=parameters, logname=None)
            di_model.simulate()
            r_i = di_model.mass_model.r[0]
            z[j, i] = r_i
    with h5py.File(simulation_name + '.hdf5', 'w') as h5file:
        h5file.create_dataset('orientation_data', data=z)
    print(f'saved to {simulation_name}.hdf5')

########################################################################################################################
# OVERVIEW PLOT OF 6 DIFFERENT E-STRENGTHS
########################################################################################################################
if plot_orientations:

    with h5py.File(simulation_name + '.hdf5', 'r') as h5file:
        data = np.array(h5file['orientation_data'])*1e3 # conversion from 1/ms to 1/s

    label = "r (Hz)"
    if voltage_view:
        delay = 1.0
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
                data[i, j] = delay_signal(v_out_hp, delay, dt) / 1000
    z_max = data.max()
    fig, axs = plt.subplots(2, 3,subplot_kw=dict(projection="polar"), figsize=(12, 10))
    for i, E_i in enumerate(E_values):
        row_idx = 0
        col_idx = i
        if i > 2:
            row_idx = 1
            col_idx -= 3

        ax = axs[row_idx, col_idx]
        z_i = data[i]
        pcm = ax.pcolormesh(theta_grid, r, z_i, shading="auto", cmap='gnuplot2', vmax=z_max)

        ax.set_thetagrids(np.arange(0, 360, 30))
        ax.set_rgrids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        if i == 1:
            fig.colorbar(pcm, label='V (µV)', orientation="horizontal")
            ax.set_rgrids([2, 4, 6, 8, 10, 12])
        ax.set_thetamax(180)
        ax.set_thetamin(0)

        ax.set_theta_zero_location("N")  # theta=0 at the top
        ax.set_theta_direction(-1)

        # ax.set_rorigin(-1.5)   # try different negative values
        ax.set_rlim(0, 12)      # make sure limits allow the shift
        # ax.set_ylabel('t in (ms)')
        ax.text(-1, 6, 't (ms)', rotation=90)
        ax.set_rlabel_position(10)
        ax.set_title(f'|E|: {E_i}V/m')
        # cbar = fig.colorbar(pcm, ax=cbar_ax, label="Intensity", orientation="horizontal")
        # plt.tight_layout()
    # plt.tight_layout()

    # fig.subplots_adjust(right=0.7)
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.15, 0.15, 0.8, 0.01])
    fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal", label=label)
    # plt.tight_layout()
    # plt.show()
    plt.savefig('CNS26_orientation.png', dpi=600)

########################################################################################################################
# SINGLE RUN OF ONE E-FIELD STRENGTH
########################################################################################################################
if single_shot:
    z = np.zeros((r_shapes[0], r_shapes[1]))
    for i in tqdm(range(n_theta)):
            theta_i = theta_deg[i]
            E_j = single_shot_E
            parameters = {'intensity': E_j, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.5,
                      'i_scale': 5.148136e-6, 'theta': theta_i, 'detrend': True,
                      'fn_session': fn_session, 'T': T, 'name': 'orientation_diw_single_run', 'dt': dt, 'min_delay': 0,
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
                                            'voltage_sigma': 4,
                                            'verbose': 0,
                                            'tqdm_disable': True}}

            di_model = DI_wave_simulation(parameters=parameters, logname=None)
            di_model.simulate()
            r_i = di_model.mass_model.r[0]
            z[i] = r_i
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
    fig, ax = plt.subplots(1, 1,subplot_kw=dict(projection="polar"), figsize=(8, 8))

    pcm = ax.pcolormesh(theta_grid, r, z, shading="auto", cmap='gnuplot2', vmax=z_max)

    ax.set_thetagrids(np.arange(0, 360, 30))
    ax.set_rgrids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_thetamax(180)
    ax.set_thetamin(0)

    ax.set_theta_zero_location("N")  # theta=0 at the top
    ax.set_theta_direction(-1)

    # ax.set_rorigin(-1.5)   # try different negative values
    ax.set_rlim(0, 12)      # make sure limits allow the shift
    # ax.set_ylabel('t in (ms)')
    ax.text(-1, 6, 't (ms)', rotation=90)
    ax.set_rlabel_position(10)
    ax.set_title(f'|E|: {single_shot_E}V/m')
        # cbar = fig.colorbar(pcm, ax=cbar_ax, label="Intensity", orientation="horizontal")
        # plt.tight_layout()
    # plt.tight_layout()

    fig.colorbar(pcm, label=label)
    plt.tight_layout()
    plt.show()

