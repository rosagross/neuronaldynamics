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
from Utils import DI_wave_test_function, nrmse, cross_correlation_align, butter_highpass_filter
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

        self.test_func_intensity = 1.5
        self.test_func_t0 = 0.2
        self.test_func_dt = 1.5
        self.test_func_width = 0.3

        self.create_convolution_plot = False
        self.save_plots = False
        self.use_gpc = True
        self.fn_session = None
        self.t_gpc = np.linspace(0, 99.81, 500)
        self.plot_align = False
        self.enable_high_pass = False
        self.test_signal_from_file = False

        if logname != None:
            self.load_from_file(logname=logname)
        elif parameters != None:
            self.parameters = parameters
        else:
            raise ValueError('Please specify paramters or logname to init class!')

        self.__dict__.update(self.parameters)

        self.t = np.arange(0, self.T, self.dt)
        bi_exp_kernel_parameters = {'tau_1': 0.2, 'tau_2': 1.7, 'tau_cond': 1, 'g_peak': 1e-4}
        init_nykamp_parameters = {'u_rest': -70, 'u_thr': -55, 'u_exc': 0, 'u_inh': -75, 'tau_mem': [12], 'tau_ref': [1.0],
                                  'delay_kernel_type': 'bi-exp', 'delay_kernel_parameters': bi_exp_kernel_parameters,
                                  'input_type': 'current', 'input_function_idx': [0, 0], 'name': self.name,
                                  'dt': self.dt, 'T': self.T, 'sparse_mat': True}

        self.create_coords()
        self.update_gpc_time()
        if self.use_gpc:
            self.load_gpc_session()
            self.grid = pygpc.RandomGrid(parameters_random=self.session.parameters_random, coords=self.coords)
            self.input_current = self.session.gpc[0].get_approximation(self.coeffs, self.grid.coords_norm) * self.i_scale
            self.input_current = self.input_current.flatten()
            self.input_current *= 1e6 # convert to µA from A
            self.input_current[np.where(self.input_current < 0)[0]] = 0
            self.input_current = np.interp(self.t, self.t_gpc, self.input_current)  # interpolate to desired time
        init_nykamp_parameters.update(self.nykamp_parameters)
        self.nykamp_parameters = init_nykamp_parameters
        self.nykamp_parameters['input_function'] = self.input_current
        self.mass_model = Nykamp_Model_1(parameters=self.nykamp_parameters)


    def simulate(self):
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
            v_out_hp = butter_highpass_filter(nmm_potential_out, cutoff=0.3, fps=int(1 / self.dt))
            hp_mean = v_out_hp.mean()
            if hp_mean > 1:
                v_out_hp -= hp_mean
            else:
                v_out_hp += hp_mean
            v_out_hp[v_out_hp < 0] = 0
            nmm_potential_out = v_out_hp

        self.get_test_signal(from_file=self.test_signal_from_file)
        di_max = np.max(self.target)
        I1_time = np.argmax(mass_model_rate) * self.dt
        # if np.max(mass_model_rate) > 0.1 and I1_time < 4:  # only scale to normalize if rate is sufficiently large
        if I1_time < 4:
            nmm_potential_scaled = nmm_potential_out / np.max(nmm_potential_out) * di_max
        else:
            nmm_potential_scaled = nmm_potential_out
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
            self.coeffs = f["coeffs"][:]

    def create_coords(self):
        self.coords = np.array([[self.theta, self.gradient, self.intensity, self.fraction_nmda, self.fraction_gaba_a,
                                 self.fraction_ex]])
    def plot_input_current(self):
        plt.plot(self.t, self.input_current*1e3, linewidth=2, c='orange')
        plt.xlabel('time in ms', fontsize=15)
        plt.ylabel('Iext in nA', fontsize=15)
        plt.show()

    def get_test_signal(self, plot=False, from_file=False):
        #TODO: extend this to different test function types eventually
        if not from_file:
            self.target = DI_wave_test_function(self.t,
                                                intensity=self.test_func_intensity,
                                                t0=self.test_func_t0,
                                                dt=self.test_func_dt,
                                                width=self.test_func_width)
        else:
            current_directory = os.path.dirname(__file__)
            data_fname = os.path.join(current_directory, 's2020_043_CNS2023.mat')
            data = scipy.io.loadmat(data_fname)
            mean_DI_waves_detrend = data['meanDIwaves_detrend']
            mean_DI_waves = data['meanDIwaves']
            t = np.array(data['times'])[0]
            self.target = np.interp(self.t, t, mean_DI_waves_detrend[:, 0])
            # plt.plot(t, mean_DI_waves_detrend[:, 0])
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
        self.error, self.difference, self.target_aligned = cross_correlation_align(x1, x2, plot=self.plot_align)


    def plot_nmm_out(self, heat_map=True, plot_input=True, save_fig=False):
        self.mass_model.plot(heat_map=heat_map, plot_input=plot_input, savefig=save_fig)

    def plot_validation(self, labels=None, save_fig=False):

        if labels == None:
            label1 = 'nykamp_potential'
            label2 = 'D-I-wave test function'
        else:
            label1, label2 = labels[0], labels[1]

        plt.plot(self.t, self.mass_model_v_out)
        plt.plot(self.t, self.target_aligned)
        plt.grid()
        plt.xlabel('t in ms')
        plt.legend([label1, label2])
        # plt.legend(['nykamp rate', 'nykamp_potential', 'D-I-wave test function'])
        plt.title(f'nrmse: {self.error:.4f}')
        if save_fig:
            plt.savefig(self.name + '_validation.png')
            plt.close()
        else:
            plt.show()
    def save_log(self, plot=True):
        log_dict = self.parameters
        log_file_name = self.name + '_log.yaml'
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


    def optimize(self):
        lower_bound = np.array([100, 0.25, 0.9, 0.2])
        upper_bound = np.array([200, 0.75, 1.0, 0.8])

        max_iter = 10
        noise_term = 0.005
        n_param = lower_bound.shape[0]
        eps = 0.01
        min_errors = []

        opt_idxs = []
        n_grid = 20
        x_vals = np.zeros((max_iter, n_grid, self.t.shape[0]))
        min_error_idxs = np.zeros(max_iter)
        parameters = np.zeros((max_iter, n_grid, lower_bound.shape[0]))
        error = np.ones((max_iter, n_grid))
        previous_min_error = 1

        for i in tqdm(range(max_iter)):
            param_values = np.zeros((n_param, n_grid))
            for j in range(n_param):
                param_values[j] = np.random.uniform(lower_bound[j], upper_bound[j], n_grid)

            for k in range(n_grid):
                parameters[i, k] = param_values[:, k]
                x = simulate(intensity=param_values[0, k], fraction_nmda=param_values[1, k],
                             fraction_gaba_a=param_values[2, k],
                             fraction_ex=param_values[3, k], y=y, idx=f'{i}_{k}')
                x_vals[i, k] = x
                error[i, k] = nrmse(y, x)

            min_error = np.nanmin(error[i])

            min_error_idx = (i, np.nanargmin(error[i]))
            min_error_idxs[i] = min_error_idx[1]
            min_errors.append(min_error)
            opt_idxs.append(min_error_idx)
            print('#########################################################################')
            print(f'error: {min_error:.5f}, at index {min_error_idx}')
            print(f'{param_values[:, min_error_idx[1]]}')
            print('#########################################################################')

            if min_error < eps:
                print(f'error: {min_error:.4f}')
                break
            if i > 0:
                previous_min_error = np.min(np.array(min_errors[:i]))
            if min_error < previous_min_error - noise_term:
                # contract region in parameter space if error was smaller
                print(f'new min error, updating parameters')

                # get new bounds for next iteration
                p_new = param_values[:, min_error_idx[1]]
                delta = upper_bound - lower_bound
                for j in range(n_param):
                    lower_bound[j] = max(lower_bound[j], p_new[j] - 0.5 * delta[j])
                    upper_bound[j] = min(upper_bound[j], p_new[j] + 0.5 * delta[j])
            else:
                print(f'error not smaller than {previous_min_error:.4f}-{noise_term}')

        plt.close()
        idx = np.argmin(error, axis=0)
        nykamp_potential = x_vals[idx[0], idx[1]]
        # print(f'optimal param: {p_new}')
        np.savetxt(X=parameters, fname='parameter_values.csv')
        np.savetxt(X=x_vals, fname='x_values.csv')

        diff = nrmse(y, nykamp_potential)
        plt.plot(t_new, nykamp_potential)
        plt.plot(t_new, y)
        plt.grid()
        plt.xlabel('t in ms')
        plt.legend(['nykamp', 'D-I-wave test function'])
        plt.title(f'nrmse: {diff:.4f}')
        plt.show()

    def optimize(self, optimizer='hierarchical', opt_params={}):

        if optimizer == 'hierarchical':
            self.__init__(parameters=opt_params)
            self.get_test_signal()
            opt_params['y'] = self.target
            opt_params['simulation_class'] = self
            opt_params['simulate'] = self.simulate
            self.optimimization_algorithm = Hierarchical_Random(parameters=opt_params)
            self.optimimization_algorithm.run()
            # params: intensity, fraction_nmda, fraction_gaba_a, fraction_ex (ei_balance)




            # idx = np.argmin(error, axis=0)
            # nykamp_potential = x_vals[idx[0], idx[1]]
            # # print(f'optimal param: {p_new}')
            # np.savetxt(X=parameters, fname='parameter_values.csv')
            # np.savetxt(X=x_vals, fname='x_values.csv')
            #
            # diff = nrmse(y, nykamp_potential)
            # plt.plot(t_new, nykamp_potential)
            # plt.plot(t_new, y)
            # plt.grid()
            # plt.xlabel('t in ms')
            # plt.legend(['nykamp', 'D-I-wave test function'])
            # plt.title(f'nrmse: {diff:.4f}')
            # plt.show()


