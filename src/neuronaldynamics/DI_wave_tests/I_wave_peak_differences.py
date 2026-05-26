import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import matplotlib
from Model.DI_wave import DI_wave_simulation
from tqdm.contrib import itertools
from Model import generate_EP
import scipy
from Utils import butter_highpass_filter, get_peak_values
matplotlib.use('TkAgg')



#TODO: make all plots also with delayed data, could also plot 2ms more to avoid loss of info
# TODO: there is also the weird hard cut again between I-wave amps, it's wrong I guess, find out where it comes from!

voltage_view = True
plot = True
simulate = False
plot_single_result = False
theta_plot = [90, 165, 0]
# E_plot= [230, 250, 380]
E_plot= [230, 255, 300]
height = 0.5 #-3

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]


E_values = np.linspace(150, 400, 100)
theta_values = np.linspace(0, 180, 100)
E_mesh, theta_mesh = np.meshgrid(E_values ,theta_values)
mesh_shapes = E_mesh.shape
z = np.zeros((mesh_shapes[0], mesh_shapes[1], t.shape[0]))
path_root = 'C:\\Users\\emueller'
# path_root = '/home/emueller'
fn_session = os.path.join(path_root, 'Downloads', 'gpc.pkl')
hdf5_path = os.path.join(path_root, '\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5')
save_file_path = os.path.join(path_root, '\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning')

simulation_name = 'E_theta_2D'
fname = os.path.join(save_file_path, simulation_name)
n_theta = theta_values.shape[0]
n_E = E_values.shape[0]

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

if plot:

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

    # load all relevant params from the file
    dt_I12 = np.zeros_like(E_mesh)
    dt_I23 = np.zeros_like(E_mesh)
    dt_I34 = np.zeros_like(E_mesh)
    dA_I12 = np.zeros_like(E_mesh)
    dA_I23 = np.zeros_like(E_mesh)
    dA_I34 = np.zeros_like(E_mesh)
    tI1 = np.zeros_like(E_mesh)
    amp_max = np.zeros_like(E_mesh)
    I_area = np.zeros_like(E_mesh)
    n_iwaves = np.zeros_like(E_mesh)

    theta_idx = np.where(theta_values > theta_plot[1])[0][0]
    E_idx = np.where(E_values > E_plot[1])[0][0]

    for i in range(theta_values.shape[0]):
        for j in range(E_values.shape[0]):
            if i==theta_idx and j==E_idx:
                r=12
            peak_values_ij = get_peak_values(t, data[i, j], find_peak_args=dict(height=height))
            n_iwaves_ij = peak_values_ij['t_delta_peaks'].shape[0]
            n_iwaves[i, j] = n_iwaves_ij
            if n_iwaves_ij < 1:
                tI1[i, j] = np.nan
                amp_max[i, j] = np.nan
                I_area[i, j] = np.nan
            else:
                tI1[i, j] = peak_values_ij['peak_1_time']
                amp_max[i, j] = peak_values_ij['peak_max_amp']
                I_area[i, j] = peak_values_ij['area']
            if n_iwaves_ij < 1:
                dt_I12[i, j] = np.nan
                dA_I12[i, j] = np.nan
            else:
                dt_I12[i, j] = peak_values_ij['t_delta_peaks'][0]
                dA_I12[i, j] = peak_values_ij['amp_delta_peaks'][0]
            if n_iwaves_ij < 2:
                dt_I23[i, j] = np.nan
                dA_I23[i, j] = np.nan
            else:
                dt_I23[i, j] = peak_values_ij['t_delta_peaks'][1]
                dA_I23[i, j] = peak_values_ij['amp_delta_peaks'][1]
            if n_iwaves_ij < 3:
                dt_I34[i, j] = np.nan
                dA_I34[i, j] = np.nan
            else:
                dt_I34[i, j] = peak_values_ij['t_delta_peaks'][1]
                dA_I34[i, j] = peak_values_ij['amp_delta_peaks'][1]

    fig, axs = plt.subplots(3, 4, figsize=(13, 9))
    z_max = data.max()


    measures = [[dt_I12, dt_I23, dt_I34], [dA_I12, dA_I23, dA_I34], [tI1, amp_max, n_iwaves]]
    measures_labels = [['delta t I1-I2 (ms)', 'delta t I2-I3 (ms)', 'delta t I3-I4 (ms)'],
                       ['delta amp I1-I2 (µV)', 'delta amp I2-I3 (µV)', 'delta amp I3-I4 (µV)'],
                       ['delay first I-wave (ms)', 'max I-wave amp (µV)', 'number of I-waves']]
    for j in range(3):
        row_idx = j
        for i in range(4):
            col_idx = i
            if i < 3:
                ax = axs[row_idx, col_idx]
                z_i = measures[j][i]
                z_i_max = z_i.max()
                z_i_min = z_i.min()
                pcm = ax.pcolormesh(E_mesh, theta_mesh, z_i, shading="auto", cmap='gnuplot2', vmax=z_max)
                # ax.set_ylabel('t in (ms)')
                ax.set_title(measures_labels[j][i])
                ax.grid(True)
                cbar = fig.colorbar(pcm)# , label=measures_labels[j][i])
                if row_idx ==2:
                    ax.set_xlabel('E (V/m)')
                if col_idx == 0:
                    ax.set_ylabel('Phi (°)')
                for l in range(len(E_plot)):
                    ax.scatter(E_plot[l], theta_plot[l], color='green')
                # ax.set_xticks([0, 2, 4, 5, 6, 7, 8, 10, 12])
            else:
                ax = axs[row_idx, col_idx]
                theta_idx = np.where(theta_values > theta_plot[j])[0][0]
                E_idx = np.where(E_values > E_plot[j])[0][0]
                signal_j = data[theta_idx, E_idx]
                peak_idxs = scipy.signal.find_peaks(signal_j,height=height)[0]
                peak_vvals = signal_j[peak_idxs]
                peak_tvals = t[peak_idxs]
                ax.plot(t, signal_j)
                ax.set_ylabel('v (µV)')
                ax.set_xlabel('t (ms)')
                ax.grid(True)
                ax.scatter(peak_tvals, peak_vvals, marker='x', c='red')
                ax.set_title(f'E = {E_plot[j]}V/m, phi = {theta_plot[j]}°')
                for l in range(len(peak_vvals)):
                    ax.text(peak_tvals[l]+0.1, peak_vvals[l]+0.1,f'{peak_tvals[l]:.1f} ms, {peak_vvals[l]:.1f} µV',
                            size=8, color='k')
    plt.tight_layout()
    plt.show()

if plot_single_result:

    with h5py.File(simulation_name + '.hdf5', 'r') as h5file:
        data = np.array(h5file['E_theta_2D'])*1e3 # conversion from 1/ms to 1/s
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
    theta_idx = np.where(theta_values > theta_plot)[0][0]
    E_idx = np.where(E_values > E_plot)[0][0]
    y = data[theta_idx, E_idx]
    plt.plot(t, y)
    plt.ylabel('v (µV)')
    plt.xlabel('t (ms)')
    plt.grid()
    plt.show()
