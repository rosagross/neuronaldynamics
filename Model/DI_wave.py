from matplotlib.lines import lineStyles

from Model.Nykamp_Model import Nykamp_Model_1
from Model.Neck import generate_EP
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pygpc
import os
import h5py
import scipy
import scipy.io
import yaml
from tqdm import tqdm
from Utils import DI_wave_test_function, nrmse, cross_correlation_align, butter_highpass_filter, detrend, delay_signal
import Model.Nykamp_Model
from Optimizers.Optimizer import *
matplotlib.use('TkAgg')

class DI_wave_simulation():

    def __init__(self, parameters=None, logname=None):

        self.name = 'di_wave_sim'
        self.i_scale = 5.148136e-9
        self.T = 20
        self.dt = 0.01
        self.dv = 0.1

        self.theta = 0  # angle of e-field [0, 180]°
        self.gradient = 0  # relative gradient of e-field [-20, 20] %/mm
        self.intensity = 200  # intensity of e-field [100, 400] V/m
        self.fraction_nmda = 0.5  # fraction of nmda synapses [0.25, 0.75]
        self.fraction_gaba_a = 0.95  # fraction of gaba_a synapses [0.9, 1.0]
        self.fraction_ex = 0.5  # fraction of exc/ihn synapses [0.2, 0.8]

        self.test_func_intensity = 1.5  # intensity (electric field magnitude) for test function
        self.test_func_t0 = 0.2  # start time of test function in ms
        self.test_func_dt = 1.5  # inter-peak time interval set in test function in ms
        self.test_func_width = 0.3  # width of peaks set in test function in ms
        self.max_shift_validation = 3 # default of 3ms max shift for comparing DI wave

        self.create_convolution_plot = False
        self.save_plots = False
        self.use_gpc = True
        self.fn_session = None
        self.t_gpc = np.linspace(0, 99.81, 500)
        self.plot_align = False
        self.plot_detrend = False
        self.enable_high_pass = False
        self.detrend = False
        self.detrend_distance = 1  # distance in ms for peaks to be considered in detrending
        self.test_signal_from_file = False
        self.error_mode = 'non-zero'
        self.min_delay = None
        self.mass_model_connectivity_matrix = None
        self.pdf_offset = 0
        self.pdf_sigma = 1
        self.pdf_weight = 1
        self.current_sigma = 10
        self.g_eext_factor = 1
        self.c_eext1_factor = 1
        self.c_eext2_factor = 1
        self.nykamp_parameters = {}
        self.file_args = None
        self.delay_signal = False
        self.delay = 2

        if logname != None:
            self.load_from_file(logname=logname)
        elif parameters != None:
            self.parameters = parameters
        else:
            raise ValueError('Please specify paramters or logname to init class!')

        self.__dict__.update(self.parameters)

        self.t = np.arange(0, self.T, self.dt)
        # higher level parameter implementation to make them available as optimization parameter
        if self.mass_model_connectivity_matrix != None:
            if type(self.mass_model_connectivity_matrix) == np.ndarray:
                if len(self.mass_model_connectivity_matrix.shape) == 1:
                    self.mass_model_connectivity_matrix = self.mass_model_connectivity_matrix[:, np.newaxis]
            elif isinstance(self.mass_model_connectivity_matrix, (int, float)):
                self.mass_model_connectivity_matrix = np.array([[self.mass_model_connectivity_matrix]])
            self.nykamp_parameters['connectivity_matrix'] = self.mass_model_connectivity_matrix
        bi_exp_kernel_parameters = {'tau_1': 0.2, 'tau_2': 1.7, 'tau_cond': 1, 'g_peak': 1e-4}
        init_nykamp_parameters = {'u_rest': -70, 'u_thr': -55, 'u_exc': 0, 'u_inh': -75, 'tau_mem': [12], 'tau_ref': [1.0],
                                  'delay_kernel_type': 'bi-exp', 'delay_kernel_parameters': bi_exp_kernel_parameters,
                                  'input_type': 'current', 'input_function_idx': [0, 0], 'name': self.name,
                                  'dt': self.dt, 'T': self.T, 'sparse_mat': True, 'g_eext_factor': self.g_eext_factor,
                                  'c_eext1_factor': self.c_eext1_factor, 'c_eext2_factor': self.c_eext2_factor,
                                  'init_pdf_offset': self.pdf_offset, 'init_pdf_sigma': self.pdf_sigma,
                                  'init_pdf_weight': self.pdf_weight, 'current_sigma': self.current_sigma}

        self.create_coords()
        self.update_gpc_time()
        if self.use_gpc:
            self.load_gpc_session()
            self.grid = pygpc.RandomGrid(parameters_random=self.session.parameters_random, coords=self.coords)
            self.input_current = self.session.gpc[0].get_approximation(self.gpc_coeffs, self.grid.coords_norm) * self.i_scale
            self.input_current = self.input_current.flatten()
            # self.input_current *= 1e6 # convert to µA from A
            self.input_current[np.where(self.input_current < 0)[0]] = 0
            self.input_current = np.interp(self.t, self.t_gpc, self.input_current)  # interpolate to desired time
        init_nykamp_parameters.update(self.nykamp_parameters)
        self.nykamp_parameters = init_nykamp_parameters
        self.nykamp_parameters['input_function'] = self.input_current
        self.mass_model = Nykamp_Model_1(parameters=self.nykamp_parameters)


    def simulate(self, r_file=None):
        """
        Simulation function that calls the neural mass model simulation function
        It then also convolves the result from the NMM, which is a rate into a voltage
        This voltage can then be filtered by a high-pass filter
        Eventually the voltage signal is stored in self.mass_model_v_out
        Finally the signal is validated against a test signal with self.validate, where an error is calculated and
        stored in self.error
        """
        if not isinstance(r_file, np.ndarray):
            self.mass_model.simulate()
            if self.save_plots:
                self.mass_model.plot(heat_map=True, plot_input=True)

            mass_model_rate = self.mass_model.r[0]
        else:
            mass_model_rate = r_file
        EP, t_EP, AP_out = generate_EP(d=0.1, plot=False, Axontype=1, dt=self.dt * 10)
        EP = -EP
        EP = EP / np.max(EP)
        EP_small = np.interp(self.t[self.t < 1.0] - 0.5, t_EP, EP)
        self.neck_kernel = EP
        self.neck_kernel_small = EP_small
        nmm_potential = scipy.signal.convolve(mass_model_rate, EP_small)
        nmm_shape = mass_model_rate.shape[0]
        nmm_potential_out = nmm_potential[:nmm_shape]

        if self.enable_high_pass:
            v_out_hp = butter_highpass_filter(nmm_potential_out, cutoff=0.05, fps=int(1 / self.dt))  # very small cutoff
            v_out_mean = nmm_potential_out.mean()
            # hp_mean = v_out_hp.mean()  # reset mean to 0
            # if hp_mean > 1:
            #     v_out_hp -= hp_mean
            # else:
            #     v_out_hp += hp_mean
            t_4ms_idx = np.where(self.t>4)[0][0]
            v_out_hp_after_4ms = v_out_hp[t_4ms_idx:]
            peaks = scipy.signal.find_peaks(-v_out_hp_after_4ms)[0]
            peaks_v = v_out_hp_after_4ms[peaks]
            v_out_hp -= np.mean(peaks_v)
            # v_out_hp += v_out_mean/8  # rescale to original height (a bit?)
            # v_out_hp[v_out_hp < 0] = 0
            nmm_potential_out = v_out_hp
        if self.detrend:
            # for find peaks in detrend: distance should be about 1ms, int(2/self.dt) as index
            nmm_potential_out = detrend(self.t, nmm_potential_out,
                                        find_peaks_args=dict(distance=int(self.detrend_distance/self.dt)),
                                        plot=self.plot_detrend, start_from_first_peak=True)

        self.get_test_signal(from_file=self.test_signal_from_file, hdf5_args=self.file_args)
        di_max = np.max(self.target)
        I1_time = np.argmax(mass_model_rate) * self.dt
        if isinstance(self.min_delay, (float, int)):
            t_idx_delay = np.where(self.t > self.min_delay)[0][0]
            potential_max = nmm_potential_out[:].max()
            nmm_potential_scaled = nmm_potential_out / potential_max * di_max
        else:
            nmm_potential_scaled = nmm_potential_out / np.max(nmm_potential_out) * di_max

        # previous version to cut out large spikes after 4ms
        # if np.max(mass_model_rate) > 0.1 and I1_time < 4:  # only scale to normalize if rate is sufficiently large
        # if I1_time < 4:
        #     nmm_potential_scaled = nmm_potential_out / np.max(nmm_potential_out) * di_max
        # else:
        #     nmm_potential_scaled = nmm_potential_out

        self.mass_model_v_out = nmm_potential_scaled

        if self.delay_signal:
            self.mass_model_v_out = delay_signal(self.mass_model_v_out, self.delay, self.dt)
        # self.plot_nmm_out()
        # self.plot_convolution()
        self.validate()
        # log


    def update_gpc_time(self):
        self.dt_gpc = np.diff(self.t_gpc)[0]
        self.T_gpc = self.t_gpc[-1] + self.dt_gpc

    def load_gpc_session(self):
        assert self.fn_session != None, 'Please provide a filename for the gpc-model!'
        self.session = pygpc.read_session(fname=self.fn_session)
        with h5py.File(os.path.splitext(self.fn_session)[0] + ".hdf5", "r") as f:
            self.gpc_coeffs = f["coeffs"][:]

    def create_coords(self):
        self.coords = np.array([[self.theta, self.gradient, self.intensity, self.fraction_nmda, self.fraction_gaba_a,
                                 self.fraction_ex]])
    def plot_input_current(self):
        plt.plot(self.t, self.input_current*1e3, linewidth=2, c='orange')
        plt.xlabel('time in ms', fontsize=15)
        plt.ylabel('Iext in nA', fontsize=15)
        plt.show()

    def get_test_signal(self, plot=False, from_file=False, fname='s2020_043_CNS2023.mat', hdf5_args=None,
                        highpass=False, hp_cutoff=1.5, plot_d_wave_detection=False):
        if self.enable_high_pass and not highpass:
            highpass=True
        if self.file_args != None and "hdf5_path" in self.file_args.keys():
            fname = self.file_args['hdf5_path']
        if not from_file:
            self.target = DI_wave_test_function(self.t,
                                                intensity=self.test_func_intensity,
                                                t0=self.test_func_t0,
                                                dt=self.test_func_dt,
                                                width=self.test_func_width)
        elif fname.split('.')[1] == 'mat':
            current_directory = os.path.dirname(__file__)
            data_fname = os.path.join(current_directory, fname)
            data = scipy.io.loadmat(data_fname)
            mean_DI_waves_detrend = data['meanDIwaves_detrend']
            mean_DI_waves = data['meanDIwaves']
            t = np.array(data['times'])[0]
            self.target = np.interp(self.t, t, mean_DI_waves_detrend[:, 0])
            # plt.plot(t, mean_DI_waves_detrend[:, 0])
        elif fname.split('.')[1] == 'hdf5' or fname.split('.')[1] == 'h5':
            hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
            data_dict = dict(orientation='PA', threshold=100, year=2020, threshold_type='RMT', channel=0, subject=0,
                             hdf5_path=hdf5_path, sigma=1)
            data_dict.update(self.file_args)
            d_wave_width = 1.8
            with h5py.File(data_dict['hdf5_path'], 'r') as h5file:
                name_h5group = h5file[data_dict['orientation']][data_dict['threshold_type']][str(data_dict['threshold'])][str(data_dict['year'])]
                name_dict = dict(name_h5group)
                name_keys = name_dict.keys()

                di_signals = []
                times = []

                subjects = []
                subject = data_dict['subject']
                if not isinstance(subject, list):
                    subject = [subject]
                for i_key, key in enumerate(name_keys):
                    if i_key in subject:
                        subject_i = name_h5group[key]

                        subjects.append(subject_i)

                        times.append(np.array(subject_i['time_short']))
                        channel_h5subgroup = subject_i
                        channel_keys = dict(subject_i).keys()
                        single_channels = []
                        channel = data_dict['channel']
                        if not isinstance(channel, list):
                            channel = [channel]
                        for i_key, key in enumerate(channel_keys):
                            if i_key in channel:
                                single_channels.append(channel_h5subgroup[key])
                        di_signals.append(single_channels)
                measurement_data_original = np.array(di_signals[0][0]['signal_short'])

            # procedure to find end and start of signal, get rid of D-wave and unwanted peaks
            t = times[0]
            detrend_thr = 0.001
            d_wave_width = 1.5
            if (data_dict['orientation'] == 'PA' and data_dict['year'] == 2020 and data_dict['threshold'] == 140 and
                    data_dict['channel'] == 0):
                idx_start = 0
                idx_end = 87
                height_d_wave = 1
                detrending = False
            elif (data_dict['orientation'] == 'PA' and data_dict['year'] == 2020 and data_dict['threshold'] == 120 and
                    data_dict['channel'] == 0):
                idx_start = 0
                idx_end = t.shape[0]
                height_d_wave = 1
                detrending = True
            elif (data_dict['orientation'] == 'PA' and data_dict['year'] == 2020 and data_dict['threshold'] == 100 and
                    data_dict['channel'] == 0):
                idx_start = 0
                idx_end = 90
                height_d_wave = 1.05
                detrending = False
            elif (data_dict['orientation'] == 'PA' and data_dict['year'] == 2007 and data_dict['threshold'] == 120 and
                    data_dict['channel'] == 0):
                idx_start = 0
                idx_end = t.shape[0]
                height_d_wave = 0.5
                detrending = False
            elif (data_dict['orientation'] == 'PA' and data_dict['year'] == 2007 and data_dict['threshold'] == 150 and
                    data_dict['channel'] == 0):
                idx_start = 0
                idx_end = t.shape[0]
                height_d_wave = 1.0
                detrending = True
                detrend_thr = 0.0002
            elif (data_dict['orientation'] == 'PA' and data_dict['year'] == 2004 and data_dict['threshold'] == 154):
                idx_start = 0
                idx_end = t.shape[0]
                height_d_wave = 0.4
                detrending = False
            elif (data_dict['orientation'] == 'PA' and data_dict['year'] == 2004 and data_dict['threshold'] == 146):
                idx_start = 0
                idx_end = t.shape[0]
                height_d_wave = 0.1
                detrending = True
                detrend_thr = 1e-4
            elif (data_dict['orientation'] == 'LM'):
                height_d_wave = measurement_data_original.max()*0.7
                idx_start = 0 #int(1/self.dt)
                idx_end = t.shape[0]
                detrending = False
            # elif (data_dict['orientation'] == 'PA' and data_dict['year'] == 2004 and data_dict['threshold'] == 150):
            #     idx_start = 0
            #     idx_end = t.shape[0]
            #     height_d_wave = 0.1
            #     detrending = False
            else:
                idx_start = 0
                idx_end = t.shape[0]
                height_d_wave = 1
                detrending = False
                d_wave_width = 2.0


            measurement_data = measurement_data_original[idx_start:idx_end].copy()

            t = t[idx_start:idx_end]
            measurement_data_filtered = scipy.ndimage.gaussian_filter1d(measurement_data, sigma=data_dict['sigma'])
            if self.detrend:
                d_wave_idx = scipy.signal.find_peaks(measurement_data, height=height_d_wave)[0][0]
            else:
                d_wave_idx = scipy.signal.find_peaks(measurement_data_filtered, height=height_d_wave)[0][0]
            if highpass:
                measurement_data_filtered = butter_highpass_filter(measurement_data_filtered,
                                                    cutoff=hp_cutoff, fps=int(1/self.dt))
                # measurement_data_filtered -= measurement_data_filtered.mean()
            t_d_wave = t[d_wave_idx]
            d_wave_start_idx = np.where(t>t_d_wave)[0][0]
            # d_wave_peak = measurement_data[d_wave_start_idx]
            # d_wave_start_idx = np.where(t>t_d_wave-0.85)[0][0]
            d_wave_end_idx = np.where(t>t_d_wave+(d_wave_width/2))[0][0]
            # measurement_data[d_wave_start_idx:d_wave_end_idx] = d_wave_peak*0.05

            if detrending and self.detrend:
                measurement_data_filtered = detrend(t, measurement_data_filtered,
                                                    find_peaks_args=dict(threshold=detrend_thr), plot=False)
                measurement_data_filtered[measurement_data_filtered<0] = 0
            measurement_data_filtered[:d_wave_end_idx] = 0
            measurement_data_filtered[-1] = 0
            if plot_d_wave_detection:
                plt.plot(t, measurement_data_filtered)
                plt.plot(t, measurement_data_original[idx_start:idx_end], alpha=0.4, color='k', linestyle='--')
                plt.xlabel('t (ms)')
                plt.ylabel('v (µV)')
                plt.title(f"{data_dict['orientation']} {data_dict['threshold']} {data_dict['year']} {data_dict['channel']+2}")
                plt.scatter(t[d_wave_idx], measurement_data_original[idx_start:idx_end][d_wave_idx], marker='x', color='r')
                plt.show()

            self.target = np.interp(self.t, t, measurement_data_filtered)
            # take caution when using this detrending



        if plot:
            plt.plot(self.t, self.target)
            plt.xlabel('time in ms')
            plt.ylabel('v in mV')
            plt.grid()
            plt.show()

    def plot_convolution(self):

        fig, ax = plt.subplots(3, 1)
        mass_model_rate = self.mass_model.r[0]
        ax[0].plot(self.t, mass_model_rate)
        ax[0].set_ylabel('DI wave potential')
        ax[1].plot(self.t[:self.neck_kernel_small.shape[0]], self.neck_kernel_small)
        ax[1].set_ylabel('Kernel')
        ax[2].plot(self.t, self.mass_model_v_out)
        ax[2].set_ylabel('DI wave rate')
        for i in range(3):
            ax[i].set_xlabel('t (ms)')
            ax[i].set_xlim([self.t[0], self.t[-1]])
        plt.show()

    def validate(self):
        x1 = self.mass_model_v_out
        x2 = self.target
        self.error, self.difference, self.target_aligned = cross_correlation_align(x1, x2, plot=self.plot_align,
                                                                                   mode=self.error_mode,
                                                                                   max_shift=int(self.max_shift_validation/self.dt))


    def plot_nmm_out(self, heat_map=True, plot_input=True, save_fig=False):
        self.mass_model.plot(heat_map=heat_map, plot_input=plot_input, savefig=save_fig)

    def plot_validation(self, labels=None, save_fig=False, fixed_ylim=False):

        if labels == None:
            label1 = 'NMM Potential'
            label2 = 'D-I-wave test function'
        else:
            label1, label2 = labels[0], labels[1]

        v_shade = self.mass_model_v_out.copy()
        abs_signal = self.target_aligned
        non_zero_mask = np.where(abs_signal > 1e-3)
        v_shade[non_zero_mask] = self.target_aligned[non_zero_mask]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t, self.mass_model_v_out)
        ax.plot(self.t, self.target_aligned)
        ax.fill_between(self.t, self.mass_model_v_out, v_shade, alpha=0.2, color='k')
        ax.grid()
        ax.set_xlabel('t in ms')
        ax.legend([label1, label2])
        if fixed_ylim:
            ax.set_ylim(-0.2*self.target.max(), 1.2*self.target.max())
        # plt.legend(['nykamp rate', 'nykamp_potential', 'D-I-wave test function'])
        ax.set_title(f'nrmse: {self.error:.4f}')
        if save_fig:
            plt.savefig(self.name + '_validation.png')
            plt.close()
        else:
            plt.show()

    def save_log(self, plot=True, log_name=None):
        """
        Function to save a log of simulation. This loc can be used to read out parameter values and internal values,
        or to recreate the simulation from the input parameters saved in the log with the from_log option in __init__
        :param plot: Bool, default True, if plots need to be saved too (no options being carried to the plot function
        as of now)
        :param log_name: str, default: None, optional name of the log_file, if this needs to be different from the
        name of the simulation object
        """
        log_dict = self.parameters.copy()
        log_dict['simulate'] = None
        if log_name == None:
            log_file_name = self.name + '_log.yaml'
        else:
            log_file_name = log_name + '_log.yaml'
        log_name = log_file_name.split('.')[0]
        if os.path.exists(log_file_name):
            if log_name[-3:] == 'log':
                log_name = log_name + '_1'
            else:
                log_name_parts = log_name.split('_')
                log_name_parts[-1] = str(int(log_name_parts[-1]) + 1)
                log_name = '_'.join(log_name_parts)
            log_file_name = log_name + '.yaml'
        self.name = log_name[:-4]
        # TODO: find a way to update the output name here
        # self.mass_model.name = self.name
        if plot:
            self.plot_nmm_out(save_fig=True)
            self.plot_validation(save_fig=True)

        print(f'saved log to: {log_file_name}')
        with open(log_file_name, 'w') as file:
            yaml.dump(log_dict, file)

    def load_from_file(self, logname):
        with open(logname, 'r') as stream:
            self.parameters = yaml.load(stream, Loader=yaml.Loader)

    def optimize(self, optimizer='hierarchical', opt_params={}):

        if optimizer == 'hierarchical':
            self.__init__(parameters=opt_params)
            self.get_test_signal()
            opt_params['y'] = self.target
            opt_params['simulation_class'] = self
            opt_params['simulate'] = self.simulate
            self.optimimization_algorithm = Hierarchical_Random(parameters=opt_params)
            self.optimimization_algorithm.run()
        elif optimizer == 'GA':
            self.__init__(parameters=opt_params)
            self.get_test_signal()
            opt_params['y'] = self.target
            opt_params['reference'] = self.target
            opt_params['simulation_class'] = self
            opt_params['simulate'] = self.simulate
            self.optimimization_algorithm = GA(parameters=opt_params)
            self.optimimization_algorithm.run()

