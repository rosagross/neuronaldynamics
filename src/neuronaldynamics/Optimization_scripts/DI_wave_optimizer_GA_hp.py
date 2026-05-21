import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')
from neuronaldynamics.Model.DI_wave import DI_wave_simulation

def main():
    T = 10
    dt = 0.01
    dv = 0.01

    # fn_session = '/home/erik/Downloads/gpc.pkl'
    # fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
    fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
    hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
    # hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
    # hdf5_path ="/home/erik/Nextcloud_Uni/TMS Neuro Projects/M1_modeling/DI_wave_data/extracted_DI_waves/DiLazarro_di_wave_data.hdf5"
    simulation_name = f'_diw_opt_chain_19_05_26_GA'
    results_path = "D:\\work\\DI-wave\\results_2026_05_19"
    # results_path = "optimization_temp"
    simulation_name = os.path.join(results_path, simulation_name)
    measurement_dict_2007_120_PA_ch3 = dict(orientation='PA', threshold=120, year=2007, hdf5_path=hdf5_path, sigma=1.0)
    print(f'name: {simulation_name}')
    parameters = {'intensity': 220, 'fraction_nmda': 0.5, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.4, 'plot_align': False,
                  'test_func_intensity': 2.0, 'test_func_t0': 0.25, 'max_shift_validation': 1.0,
                  'test_signal_from_file': True, 'i_scale': 5.148136e-9, 'error_mode': 'a',
                  'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'enable_high_pass': True,
                  'detrend': False, 'delay_signal': True, 'delay': 0.9,
                  'file_args': measurement_dict_2007_120_PA_ch3,
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
    di_model.get_test_signal()

    model_parameters = ['theta', 'intensity', 'fraction_nmda', 'fraction_gaba_a', 'fraction_ex', 'pdf_offset', 'pdf_sigma',
                         'current_sigma']
    model_parameter_bounds = [[0, 180], [200, 400], [0.25, 0.75], [0.9, 1.0], [0.2, 0.8], [0, 12], [0.1, 15], [0, 15]]

    # ['best after gradient: [96.89784643] at param set [[ 60.93360198 202.03708798   0.57166902   0.99426977   0.52796225\n    3.78881251   2.5311716    2.2537436 ]]']
    opt_parameters = parameters.copy()
    opt_parameters['optimizer'] = 'GA'
    opt_parameters['n_cpus'] = 2
    opt_parameters['serial_computation'] = True
    opt_parameters['results_folder'] = results_path
    opt_parameters['reference'] = di_model.target
    opt_parameters['tolerance'] =  5
    opt_parameters['single_run_tol'] = 0.1
    opt_parameters['n_iter'] = 20
    opt_parameters['N1'] = 5
    opt_parameters['N2'] = 5
    opt_parameters['N3'] = 5
    opt_parameters['model_parameters'] = model_parameters
    opt_parameters['simulation_class'] = di_model
    opt_parameters['simulate'] = di_model.simulate
    opt_parameters['bounds'] = model_parameter_bounds
    opt_parameters['x_out'] = 'mass_model_v_out'
    opt_parameters['nykamp_parameters']['tqdm_disable'] = True
    opt_parameters['save_results'] = True

    di_model.optimize(optimizer='GA', opt_params=opt_parameters)
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
    di_model.name = simulation_name # set name to simulation name to find it again
    di_model.plot_validation(save_fig=True)
    np.savetxt(f'{simulation_name}_params.csv', opt_params)
    os.remove(simulation_name + '.hdf5')
    error_test = di_model.error

if __name__ == "__main__":
    main()