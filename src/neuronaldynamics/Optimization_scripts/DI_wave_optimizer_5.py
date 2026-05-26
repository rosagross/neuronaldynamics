import numpy as np
import matplotlib
from neuronaldynamics.Model.DI_wave import DI_wave_simulation

if __name__ == "__main__":
    matplotlib.use('TkAgg')
    T = 10
    dt = 0.01
    dv = 0.01
    # fn_session = '/home/erik/Downloads/gpc.pkl'
    fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
    # fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'

    # hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
    hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
    simulation_name = 'diw_opt_hu_20_01_26'
    measurement_dict_2020_140_PA_ch3 = dict(orientation='PA', threshold=140, year=2020, hdf5_path=hdf5_path, sigma=1.0)
    measurement_dict_2020_100_PA_ch3 = dict(orientation='PA', threshold=100, year=2020, hdf5_path=hdf5_path, sigma=1.0)

    parameters = {'intensity': 220, 'fraction_nmda': 0.5, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.4, 'plot_align': False,
                  'test_func_intensity': 2.0, 'test_func_t0': 0.25, 'max_shift_validation': 4,
                  'test_signal_from_file': True, 'i_scale': 5.148136e-6, 'error_mode': 'a',
                  'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'enable_high_pass': False,
                  'detrend': True,
                  'file_args': measurement_dict_2020_140_PA_ch3,
                  'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
                                        'tau_ref': [0],
                                        'tau_mem': [12],
                                        'dv': dv,
                                        'init_pdf_weight': 0,
                                        'static_noise': True,
                                        'delay_kernel_type': 'alpha',
                                        'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3},
                                        'input_type': 'stochastic-current',
                                        'solver': 'hu-2021'}}
    di_model = DI_wave_simulation(parameters=parameters, logname=None)


    model_parameters = ['intensity', 'fraction_nmda', 'fraction_gaba_a', 'fraction_ex', 'pdf_offset', 'pdf_sigma',
                         'current_sigma']
    model_parameter_bounds = [[200, 400], [0.25, 0.75], [0.9, 1.0], [0.5, 0.8], [0, 12], [0.01, 5], [0, 5]]

    opt_parameters = parameters.copy()
    opt_parameters['optimizer'] = 'hierarchical'
    opt_parameters['eps'] = 0.05
    opt_parameters['max_iter'] = 3
    opt_parameters['n_grid'] = 300
    opt_parameters['model_parameters'] = model_parameters
    opt_parameters['bounds'] = model_parameter_bounds
    opt_parameters['x_out'] = 'mass_model_v_out'
    opt_parameters['nykamp_parameters']['tqdm_disable'] = True
    opt_parameters['save_results'] = True

    di_model.optimize(opt_params=opt_parameters)
    opt_params = di_model.optimimization_algorithm.optimum
    print(f'optimal params recovered: {opt_params}')

    errors = di_model.optimimization_algorithm.error
    min_error = errors.min()
    print(f'min error: {min_error:.4f}')
    min_error_idx = np.unravel_index(np.argmin(errors, axis=None), errors.shape)
    best_x = di_model.optimimization_algorithm.x_vals[min_error_idx]
    # plt.plot(best_x)
    di_model.mass_model_v_out = best_x
    di_model.validate()
    di_model.plot_validation()
    error_test = di_model.error