from Nykamp_Model import Nykamp_Model_1
from Neck import generate_EP
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pygpc
import os
import h5py
import scipy
from tqdm import tqdm
from Utils import DI_wave_test_function, nrmse
matplotlib.use('TkAgg')

class DI_wave_simulation():

    def __init__(self, parameters):

        self.name = 'di_wave_sim'
        self.i_scale = 5.148136e-9
        self.T = 20
        self.dt = 0.01
        self.t = np.arange(0, self.T, self.dt)
        self.dv = 0.1

        self.theta = 0  # angle of e-field [0, 180]°
        self.gradient = 0  # relative gradient of e-field [-20, 20] %/mm
        self.intensity = 200  # intensity of e-field [100, 400] V/m
        self.fraction_nmda = 0.5  # fraction of nmda synapses [0.25, 0.75]
        self.fraction_gaba_a = 0.95  # fraction of gaba_a synapses [0.9, 1.0]
        self.fraction_ex = 0.5  # fraction of exc/ihn synapses [0.2, 0.8]
        self.create_coords()

        self.test_func_intensity = 1.5
        self.test_func_t0 = 0.2
        self.test_func_dt = 1.5
        self.test_func_width = 0.3

        self.plot_convolution = False
        self.save_plots = False
        self.use_gpc = True
        self.fn_session = None
        self.t_gpc = np.linspace(0, 99.81, 500)
        self.update_gpc_time()
        if self.use_gpc:
            self.load_gpc_session()
            self.grid = pygpc.RandomGrid(parameters_random=self.session.parameters_random, coords=self.coords)
            self.input_current = self.session.gpc[0].get_approximation(self.coeffs, self.grid.coords_norm) * self.i_scale
            self.input_current = self.input_current.flatten()
            self.input_current *= 1e6 # convert to µA from A
            self.input_current = np.interp(self.t, self.t_gpc, self.input_current)  # interpolate to diesired time
        bi_exp_kernel_parameters = {'tau_1': 0.2, 'tau_2': 1.7, 'tau_cond': 1, 'g_peak': 1e-4}
        self.nykamp_parameters = {'u_rest': -70, 'u_thr': -55, 'u_exc': 0, 'u_inh':-75, 'tau_mem':12, 'tau_ref':1.0,
                                  'delay_kernel_type': 'bi_exp', 'delay_kernel_parameters': bi_exp_kernel_parameters,
                                  'input_type': 'current', 'input_function': self.input_current, 'name':self.name}
        self.mass_model = Nykamp_Model_1(parameters=self.nykamp_parameters)


    def simulate(self):
        self.mass_model.simulate()
        if self.save_plots:
            self.mass_model.plot(heat_map=True, plot_input=True)
        self.mass_model.clean()

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
        di_max = np.max(self.target)
        nmm_potential_scaled = nmm_potential_out / np.max(nmm_potential_out) * di_max
        self.mass_model_v_out = nmm_potential_scaled

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
        plt.plot(self.t, self.input_current)
        plt.xlabel('time in ms')
        plt.ylabel('Iext in A')
        plt.show()

    def get_test_signal(self, plot=False):
        #TODO: extend this to different test function types eventually
        self.target = DI_wave_test_function(self.t,
                                            intensity=self.test_func_intensity,
                                            t0=self.test_func_t0,
                                            dt=self.test_func_dt,
                                            width=self.test_func_width)
        if plot:
            plt.plot(self.t, self.target)
            plt.xlabel('time in ms')
            plt.ylabel('firing rate test function')
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
