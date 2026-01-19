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
from Utils import DI_wave_test_function, nrmse, cross_correlation_align, butter_highpass_filter, detrend
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


    def simulate(self):
        """
        Simulation function that calls the neural mass model simulation function
        It then also convolves the result from the NMM, which is a rate into a voltage
        This voltage can then be filtered by a high-pass filter
        Eventually the voltage signal is stored in self.mass_model_v_out
        Finally the signal is validated against a test signal with self.validate, where an error is calculated and
        stored in self.error
        """
        self.mass_model.simulate()
        if self.save_plots:
            self.mass_model.plot(heat_map=True, plot_input=True)

        mass_model_rate = self.mass_model.r[0]
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

    def get_test_signal(self, plot=False, from_file=False, fname='s2020_043_CNS2023.mat', hdf5_args=None):
        #TODO: extend this to different test function types eventually
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
            data_dict = dict(orientation='PM', threshold=100, year=2020, threshold_type='RMT', channel=0, subject=0,
                             hdf5_path=hdf5_path)
            data_dict.update(self.file_args)
            with h5py.File('.hdf5', 'r') as h5file:
                name_keys = h5py[data_dict['orientation']][data_dict['threshold_type']][data_dict['threshold']][data_dict['year']].keys()
                di_waves = h5py[data_dict['orientation']][data_dict['threshold_type']][data_dict['threshold']][data_dict['year']][name_keys[data_dict['subject']]]

            self.target = np.interp(self.t, t, mean_DI_waves_detrend[:, 0])
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

