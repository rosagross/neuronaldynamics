import numpy as np
import os
import matplotlib
import h5py
from Model.DI_wave import DI_wave_simulation

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
    simulation_name = '_diw_opt_hu_20_01_26'
    measurement_dict_2020_140_PA_ch3 = dict(orientation='PA', threshold=140, year=2020, hdf5_path=hdf5_path, sigma=1.0)
    measurement_dict_2020_100_PA_ch3 = dict(orientation='PA', threshold=100, year=2020, hdf5_path=hdf5_path, sigma=1.0)
    measurement_dict_2020_120_PA_ch3 = dict(orientation='PA', threshold=120, year=2020, hdf5_path=hdf5_path, sigma=1.0)
    measurement_dict_2013_110_PA_ch2 = dict(orientation='PA', threshold=110, year=2013, hdf5_path=hdf5_path, sigma=1.0,
                                            channel=0)
    measurement_dict_2013_110_PA_ch3 = dict(orientation='PA', threshold=110, year=2013, hdf5_path=hdf5_path, sigma=1.0,
                                            channel=1)
    measurement_dict_207_120_PA_ch3 = dict(orientation='PA', threshold=120, year=2007, hdf5_path=hdf5_path, sigma=1.0)
    measurement_dict_207_150_PA_ch3 = dict(orientation='PA', threshold=150, year=2007, hdf5_path=hdf5_path, sigma=1.0)

    data_dicts = [measurement_dict_2020_140_PA_ch3,
                  measurement_dict_2020_100_PA_ch3,
                  measurement_dict_2020_120_PA_ch3,
                  measurement_dict_2013_110_PA_ch2,
                  measurement_dict_2013_110_PA_ch3,
                  measurement_dict_207_120_PA_ch3,
                  measurement_dict_207_150_PA_ch3]

    results_path = hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_model\\run_2026_01_21"
    for i_dict, dict in enumerate(data_dicts):
        print(f'data dict #{i_dict}: {dict} \n')
        simulation_name = f'_diw_opt_chain_20_01_26_no_{i_dict}'
        parameters = {'intensity': 220, 'fraction_nmda': 0.5, 'fraction_gaba_a': 0.95, 'fraction_ex': 0.4,
                      'plot_align': False,
                      'test_func_intensity': 2.0, 'test_func_t0': 0.25, 'max_shift_validation': 4,
                      'test_signal_from_file': True, 'i_scale': 5.148136e-6, 'error_mode': 'a',
                      'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'enable_high_pass': False,
                      'detrend': True,
                      'file_args': dict,
                      'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
                                            'tau_ref': [0],
                                            'tau_mem': [12],
                                            'dv': dv,
                                            'init_pdf_weight': 0,
                                            'static_noise': True,
                                            'delay_kernel_type': 'alpha',
                                            'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1 / 3},
                                            'input_type': 'stochastic-current',
                                            'solver': 'hu-2021'}}
        di_model = DI_wave_simulation(parameters=parameters, logname=None)
        di_model.get_test_signal(plot=False, from_file=True, hdf5_args=di_model.file_args)
        h5fname = os.path.join(results_path, simulation_name)
        with h5py.File(h5fname+'.hdf5', 'r') as h5file:
            r_hdf5 = np.array(h5file['r'])[0]
        di_model.simulate(r_hdf5)
        di_model.validate()
        di_model.plot_validation()