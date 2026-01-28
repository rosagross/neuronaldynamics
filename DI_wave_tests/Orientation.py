import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
matplotlib.use('TkAgg')

plot_nykamp_basic = False
plot_di_model = True

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]


thetas = np.linspace(0, np.pi, 100)
r, theta_grid = np.meshgrid(t, thetas)
z = np.zeros_like(r)

# fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'

hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
# hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"

simulation_name = 'diw_orientation_test'

for i, theta_i in enumerate(thetas):
    theta_radians = theta_i / (2 * np.pi) * 360
    parameters = {'intensity': 350, 'fraction_nmda': 0.61, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.5,
                  'i_scale': 5.148136e-6, 'theta': theta_radians, 'detrend': True,
                  'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'mind_delay': 0,
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
                                        'verbose': 1,
                                        'tqdm_disable': True}}

    di_model = DI_wave_simulation(parameters=parameters, logname=None)
    di_model.simulate()
    r_i = di_model.mass_model.r[0]
    z[i] = r_i


# z = np.sin(r) + 1 - (theta/(2 * np.pi))

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

pcm = ax.pcolormesh(theta_grid, r, z, shading="auto", cmap='gnuplot2')

ax.set_thetagrids(np.arange(0, 360, 30))
ax.set_rgrids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
ax.set_thetamax(180)
ax.set_thetamin(0)

ax.set_theta_zero_location("N")  # theta=0 at the top
ax.set_theta_direction(-1)

# ax.set_rorigin(-1.5)   # try different negative values
ax.set_rlim(0, 12)      # make sure limits allow the shift
# ax.set_ylabel('t in (ms)')
ax.text(-0.5, 6, 't (ms)', rotation=90)
ax.set_rlabel_position(10)
fig.colorbar(pcm, ax=ax, label="Intensity")
plt.tight_layout()
plt.show()