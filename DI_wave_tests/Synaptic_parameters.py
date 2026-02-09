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
from Utils import butter_highpass_filter
matplotlib.use('TkAgg')

voltage_view = True
plot_params = True
simulate = False

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]


p_values = np.linspace(0.2, 0.8, 100)
theta_values = [k*30 for k in range(7)]
p_mesh, t_mesh = np.meshgrid(t, p_values)
mesh_shapes = p_mesh.shape
z = np.zeros((mesh_shapes[0], mesh_shapes[1]))

# path_root = '/home/erik/Downloads/'
# path_root = 'C:\\Users\\emueller'
path_root = 'C:\\Users\\User'

# fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
fn_session = os.path.join(path_root, 'Downloads', 'gpc.pkl')
hdf5_path = os.path.join(path_root, '\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5')
save_file_path = os.path.join(path_root, '\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning')

# hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
# hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"

# save_file_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning"
# save_file_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI-wave_model\\orientation_tuning"
simulation_name = 'diw_param_test'
fname = os.path.join(save_file_path, simulation_name)

# simulate
if simulate:

    for i in tqdm(range(p_values.shape[0])):
        parameters = {'intensity': 300, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': p_values[i],
                  'i_scale': 5.148136e-6, 'theta': 0, 'detrend': True,
                  'fn_session': fn_session, 'T': T, 'name': 'param_diw_single_run', 'dt': dt, 'min_delay': 0,
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
        z[i] = r_i
    with h5py.File(simulation_name + '.hdf5', 'w') as h5file:
        h5file.create_dataset('orientation_data', data=z)
    print(f'saved to {simulation_name}.hdf5')

# plot
if plot_params:
    with h5py.File(simulation_name + '.hdf5', 'r') as h5file:
        data = np.array(h5file['orientation_data'])*1e3 # conversion from 1/ms to 1/s
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
    ax.set_title(f'Variation of exc/inh ratio')
    ax.set_ylabel('Exc/Ihn ratio')
    ax.set_xlabel('t (ms)')
    ax.grid(True)
    ax.set_xticks([0, 2, 4, 5, 6, 7, 8, 10, 12])
    plt.tight_layout()
    fig.colorbar(pcm, label=label)
    plt.show()