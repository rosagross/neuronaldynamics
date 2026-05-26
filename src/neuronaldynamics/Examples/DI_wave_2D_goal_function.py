import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.DI_wave import DI_wave_simulation
import h5py
from tqdm import tqdm
matplotlib.use('TkAgg')

# fn_session = '/home/erik/Downloads/gpc.pkl'
fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
n = 100
intensities = np.linspace(150, 400, n)
connectivities = np.linspace(0, 50, n)
errors = np.zeros((n, n))
differences = np.zeros((n, n))

#
for i, _ in enumerate(tqdm(range(n))):
    for j in range(n):
        parameters = {'intensity': intensities[i], 'fraction_nmda': 0.70, 'fraction_gaba_a': 0.98, 'fraction_ex': 0.70,
                      'plot_align': False, 'test_func_intensity': 2.5, 'test_func_t0': 0.35,
                      'fn_session': fn_session, 'T': 8, 'dt': 0.02, 'enable_high_pass': False,
                      'nykamp_parameters': {'connectivity_matrix': np.array([[connectivities[j]]]),
                                            'tau_ref': [1.5],
                                            'tau_mem': [12],
                                            'input_type': 'current',
                                            'init_pdf_sigma': 1.0,
                                            'tqdm_disable': True}}
        di_model = DI_wave_simulation(parameters=parameters, logname=None)
        di_model.simulate()
        errors[i, j] = di_model.error
        differences[i, j] = di_model.difference.mean()/di_model.target.max()
        di_model.mass_model.clean()

### updated nrmse selection ####

with h5py.File('2D_sweep_data_1.hdf5', 'w') as h5file:
    h5file.create_dataset('intensities', data=intensities)
    h5file.create_dataset('connectivities', data=connectivities)
    h5file.create_dataset('errors', data=errors)
    h5file.create_dataset('differences', data=differences)

with h5py.File('2D_sweep_data_1.hdf5', 'r') as h5file:
    intensities = np.array(h5file['intensities'])
    connectivities = np.array(h5file['connectivities'])
    errors = np.array(h5file['errors'])
    differences = np.array(h5file['differences'])

fig = plt.figure()
X, Y = np.meshgrid(intensities, connectivities)
ax = fig.add_subplot(111)
z_limits = (np.abs(errors).min(), np.abs(errors).max())
z_min, z_max = z_limits
c = ax.pcolormesh(X, Y, errors, cmap='magma', vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
ax.set_ylabel('Connectivity')
ax.set_xlabel('Intensity (V/m)')
ax.set_title('NRMSE against test function')
plt.show()
