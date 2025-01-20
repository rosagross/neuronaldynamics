"""
Code from Richard Naud:
https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html
Modified by Erik Müller

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import h5py
from scipy.special import factorial
from scipy.stats import gamma
import random
from tqdm import tqdm
from Utils import raster, nrmse, time_bin, divide_axis
from scipy.ndimage import gaussian_filter1d
import time

class LIF_population():

    def __init__(self, **kwargs):
        # Set parameters
        self.default_pars(**kwargs)
        self.Iinj = None
        self.get_alpha_kernel()
        # TODO: properly init with if ... is none ... is 'lif'
        self.fname = 'lif'
        self.population_type = ['exc']


    def default_pars(self, **kwargs):
        #TODO: replace this with proper init
        """
        Function that sets default parameters
        :param kwargs: arguments that may contain defaults
            - V_th: threshold voltage
            - V_reset: reset voltage
            - tau_m: membrane time constant
            - g_L: leak conductance
            - V_init: initial potential
            - E_L: leak reversal potential
            - tref: refractory time
            - T: simulation time
            - dt: time step
            - weights: connectivity matrix
            - Iext: external current
            - tau_alpha: time constant of alpha kernel
            - n_alpha: n constant of alpha kernel
            - n_neurons: number of neurons in population
            - verbose: value to identify output
        """
        self.pars = {}

        # typical neuron parameters# arranged for nykamp model
        self.pars['V_th'] = -55.  # spike threshold [mV]
        self.pars['V_reset'] = -65.  # reset potential [mV]
        self.pars['tau_m'] = 20.  # membrane time constant [ms]
        self.pars['g_L'] = 10.  # leak conductance [nS]
        self.pars['V_init'] = -65.  # initial potential [mV]
        self.pars['E_L'] = -65.  # leak reversal potential [mV]
        self.pars['tref'] = 3.  # refractory time (ms)

        # simulation parameters #
        self.pars['T'] = 400.  # Total duration of simulation [ms]
        self.pars['dt'] = .1  # Simulation time step [ms]
        self.pars['weights'] = None
        self.pars['Iext'] = None

        self.pars['tau_alpha'] = 1/3 # parameters from Nykamp 2000 here
        self.pars['n_alpha'] = 9
        self.pars['n_neurons'] = 1
        self.pars['verbose'] = 0

        # external parameters if any #
        for k in kwargs:
            self.pars[k] = kwargs[k]

        self.t = np.arange(0, self.pars['T'], self.pars['dt'])  # Vector of discretized time points [ms]
        self.V_th, self.V_reset = self.pars['V_th'], self.pars['V_reset']
        self.tau_m, self.g_L = self.pars['tau_m'], self.pars['g_L']
        self.V_init, self.E_L = self.pars['V_init'], self.pars['E_L']
        self.dt = self.pars['dt']
        self.Lt = self.t.size
        self.T = self.pars['T']
        self.tref = self.pars['tref']
        self.Iext = self.pars['Iext']
        self.weights = self.pars['weights']
        self.n_neurons = self.pars['n_neurons']
        self.tau_alpha = self.pars['tau_alpha']
        self.n_alpha = self.pars['n_alpha']
        self.verbose = self.pars['verbose']

    def run(self, Iinj=None, stop=False, custom_i=True, no_weighting=False):
        """
        Simulate the LIF dynamics with external input current

        Args:
        pars       : parameter dictionary
        Iinj       : input current [pA]. The injected current here can be a value
                     or an array
        stop       : boolean. If True, use a current pulse

        Returns:
        rec_v      : membrane potential
        rec_sp     : spike times

        """

        if self.verbose > 0:
            t1 = time.time()

        if self.Iinj is not None:
            Iinj = self.Iinj
        else:
            Iinj = np.zeros((self.n_neurons, self.t.shape[0]))

        if self.Iext is not None:
            if len(self.Iext.shape) == 1:
                Iext_shape_init = self.Iext.shape[0]
                self.Iext = self.Iext.repeat(self.n_neurons).reshape(Iext_shape_init, self.n_neurons).T


        # Initialize voltage
        self.v = np.zeros((self.n_neurons, self.Lt))
        self.t_spikes = np.zeros_like(self.v)
        Iin = np.zeros_like(self.v)
        # randomization hard coded here
        self.v[:, 0] = self.V_init + np.random.normal(0, 0.7, self.v.shape[0])
        tr = np.zeros(self.n_neurons) # the count for refractory duration
        t_last_spike = np.zeros(self.n_neurons)
        self.r = np.zeros_like(self.v)

        if not custom_i:

            # Set current time course
            Iinj = Iinj * np.ones(self.Lt)



        # If current pulse, set beginning and end to 0
        if stop:
          Iinj[:int(len(Iinj) / 2) - 1000] = 0
          Iinj[int(len(Iinj) / 2) + 1000:] = 0

        # Loop over time
        self.rec_spikes = []

        conv_shape = self.v[0].shape[0] + self.alpha.shape[0] - 1
        input = np.zeros((self.n_neurons, conv_shape))

        # record spike times
        # create list of lists with n_neurons as fisrt dimension
        for _ in range(self.n_neurons):
            self.rec_spikes.append([])

        if self.verbose > 0:
            t2 = time.time()
            print(f'set-up: {t2-t1:.4f}s')

        for it in tqdm(range(self.Lt - 1), f'simulating network for {self.Lt - 1} time steps'):
            for i in range(self.n_neurons):

                if tr[i] > 0:  # check if in refractory period
                    self.v[i, it] = self.V_reset  # set voltage to reset
                    tr[i] = tr[i] - 1  # reduce running counter of refractory period

                elif self.v[i, it] >= self.V_th:  # if voltage over threshold
                    self.rec_spikes[i].append(it)  # record spike event
                    self.t_spikes[i, it] = 1
                    self.r[i, int(t_last_spike[i]):it] = 1000 / (
                          it - t_last_spike[i]) * self.dt  # times 1000 for conversion 1/ms -> Hz
                    t_last_spike[i] = it
                    self.v[i, it] = self.V_reset  # reset voltage
                    tr[i] = self.tref / self.dt  # set refractory time

            if self.verbose > 0:
                t_conv_1 = time.time()
            if self.n_neurons > 1:
                # problem with this version: keeps the dimension fixed while np.convolve extends for overshoot
                # test = np.einsum('i,kl->kl', self.alpha, np.sum(self.weights[:, i] * np.reshape(np.repeat(self.t_spikes, 50), (5000, 50, 50)), axis=1)).T
                # for i in range(self.n_neurons):
                #     # get input from other neurons
                #         if it > 0:
                #               # this is a bottle neck, the faster implementation is below
                #               input[i, :] = np.convolve(np.sum(self.weights[:, i] * self.t_spikes.T, axis=1), self.alpha)
                #               # connectivity weight is parameter that scales the current here
                #               # input = np.convolve(np.sum(self.weights[:, i] * self.t_spikes.T, axis=1), self.alpha)
                if no_weighting:
                    for k in range(self.n_neurons):
                        for l in range(len(self.rec_spikes[k])):
                            input[:, self.rec_spikes[k][l]: self.rec_spikes[k][l] + self.alpha.shape[0]] += self.alpha

                else:
                    # reshape weights for alpha kernel
                    weights_repeat = self.weights.repeat(self.alpha.shape[0]).reshape(
                        (self.weights.shape[0], self.weights.shape[1], self.alpha.shape[0]))
                    for k in range(self.n_neurons):
                        for l in range(len(self.rec_spikes[k])):
                            kernel_idxs = self.rec_spikes[k][l], self.rec_spikes[k][l] + self.alpha.shape[0]
                            input[:, kernel_idxs[0]:kernel_idxs[1]] += weights_repeat[k, :, :] * self.alpha

                Iin[:, it] = Iinj[:, it] + input[:, it] + self.Iext[:, it]
            else:
                Iin = Iinj + self.Iext

            if self.verbose > 0:
                t_conv_2 = time.time()
                print(f'convolve: {t_conv_2 - t_conv_1:.4f}s')
                t_sim_1 = t_conv_2

            # Calculate the increment of the membrane potential
            dv = (-(self.v[:, it] - self.E_L) + Iin[:, it] / self.g_L) * (self.dt / self.tau_m)

            # Update the membrane potential
            self.v[:, it + 1] = self.v[:, it] + dv

            if self.verbose > 0 :
                t_sim_2 = time.time()
                print(f'simulate: {t_sim_2 - t_sim_1:.4f}s')

        for i in range(self.n_neurons):
            # Get spike times in ms
            self.rec_spikes[i] = np.array(self.rec_spikes[i]) * self.dt

        spike_sum = np.sum(self.t_spikes, axis=0)

        with h5py.File(self.fname + '.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=self.t)
            h5file.create_dataset('v', data=self.v)
            h5file.create_dataset('r', data=spike_sum)
            h5file.create_dataset('p_types', data=self.population_type)

    def get_alpha_kernel(self):
        self.t_alpha = self.t[self.t < 10]
        self.alpha = np.exp(-self.t_alpha/self.tau_alpha) / (self.tau_alpha * factorial(self.n_alpha-1)) *\
                (self.t_alpha/self.tau_alpha)**(self.n_alpha-1)
        self.alpha = self.alpha/np.trapz(self.alpha, dx=self.dt)


    def plot_volt_trace(self, idx=0):
      """
      Plot trajetory of membrane potential for a single neuron

      Expects:
      pars   : parameter dictionary
      v      : volt trajetory
      sp     : spike train

      Returns:
      figure of the membrane potential trajetory for a single neuron
      """

      V_th = self.pars['V_th']
      dt, range_t = self.pars['dt'], self.t
      if self.rec_spikes[idx].size:
        sp_num = (self.rec_spikes[idx] / dt).astype(int) - 1
        self.v[idx, sp_num] += 20  # draw nicer spikes

      plt.plot(self.t, self.v[idx], 'b')
      plt.axhline(V_th, 0, 1, color='k', ls='--')
      plt.xlabel('Time (ms)')
      plt.ylabel('V (mV)')
      plt.legend(['Membrane\npotential', r'Threshold V$_{\mathrm{th}}$'],
                 loc=[1.05, 0.75])
      plt.ylim([-80, -40])
      plt.grid()
      plt.tight_layout()
      plt.show()

    def gen_poisson_spikes_input(self, i_max=300, rate=1, mu=0.008, coeff_of_var=0.5, t_start=0.0, t_end=None,
                                 delay=True):
        """
        Generate spike times and currents for a neuron with a time-dependent firing rate using an inhomogeneous Poisson
         process.
        modified from:
        https://medium.com/@baxterbarlow/poisson-spike-generators-stochastic-theory-to-python-code-a76f8cc7cc32

        Parameters:
        imax (float): Max value if I which is used as scaling factor for random sampling
        rate (float): Firing rate at time t (spikes per second).

        """

        scale = (coeff_of_var * mu) ** 2 / mu
        gamma_pdf = gamma(a=coeff_of_var ** (-2), loc=0, scale=scale)

        if not callable(rate):
            rate_copy = rate
            rate = lambda x : rate_copy

        if t_end is None:
            t_end = self.T

        ts = np.arange(0, self.T, self.dt)
        i_s = np.zeros((self.n_neurons, ts.shape[0]))
        for j in tqdm(range(self.n_neurons), f'creating background activity for {self.n_neurons} neurons'):
            t_last_spike = 0
            for i, t_i in enumerate(ts):

                if i == 0:
                    # TODO:
                    #  is need to be gamma distributed according to paper
                    interval = -np.log(np.random.rand()) / (rate(t_i)+1e-10)

                if t_i - t_last_spike > interval:
                    sign = [-1,1][random.randrange(2)]
                    i_s[j, i] = gamma_pdf.rvs(size=1) * i_max # * sign
                    t_last_spike = t_i
                    # amp = gamma.rvs(size=1)
                    interval = -np.log(np.random.rand()) / (rate(t_i) + 1e-10)

            if delay:
                i_shape = i_s[j].shape[0] + self.alpha.shape[0] - 1
                i_delayed = np.zeros(i_shape)
                idxs = np.where(i_s[j] > 0)[0]
                for i_idx, idx in enumerate(idxs):
                    i_delayed[idx:idx + self.alpha.shape[0]] += self.alpha * i_s[j, idx]
                i_s[j] = i_delayed[:ts.shape[0]]

        i_s[:, int(t_start / self.dt)] = 0
        i_s[:, int(t_end / self.dt):] = 0
        self.Iinj = i_s

    def raster_plot(self, idxs=None, color='k'):
        if idxs is not None:
            ax = raster(self.rec_spikes[idxs], color=color)
        else:
            ax = raster(self.rec_spikes, color=color)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('# neurons')
        ax.set_title('Spike raster plot')
        plt.show()

    def plot_firing_rate(self, size=(8, 6), bin_size=10, smoothing=False, sigma=2):
        fig = plt.figure(figsize=size)

        blu = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

        spike_sum = np.sum(self.t_spikes, axis=0)
        spike_hist = time_bin(spike_sum, bin_size=bin_size)
        if smoothing:
            spike_sum = gaussian_filter1d(spike_sum, sigma=sigma)
        ax = fig.add_subplot(1, 1, 1)
        # dt factor for scaling to per ms
        # 1000 factor for conversion from ms to s (kHz to Hz)
        time_scaling_factor = self.dt**-1 * 1000

        # ax.plot(self.t, spike_sum_gauss*self.dt**-1 * 1000, c=blu, linewidth=2)
        t = np.arange(0, self.T, self.dt)
        ax.plot(spike_sum*time_scaling_factor, c=blu, linewidth=2)
        ax.stairs(spike_hist * time_scaling_factor, fill=True, alpha=0.5)
        ax.set_ylabel('r in spikes/second')
        plt.tight_layout()
        ax.set_xlabel('time in ms')

        divide_axis(ax, self.dt**-1, set_int=True)
        plt.tight_layout()
        plt.show()

    def plot_voltage_hist(self, times, size=(8, 8)):
        fig = plt.figure(figsize=size)


        for n, time in enumerate(times):
            ax = fig.add_subplot(len(times), 1, int(n + 1))
            ax.hist(self.v[:, time], bins=100, density=True, alpha=0.7)
            ax.set_title(f't = {time*self.dt}ms')
            ax.set_xlabel(f'V in mV')

        plt.tight_layout()
        ax.set_xlabel('V in mv')
        plt.show()

    def plot_populations(self, plot_idxs=None, bins=100, cutoff=None, smoothing=False, sigma=2, hide_refractory=True):

        with h5py.File(self.fname + '.hdf5', 'r') as h5file:

            t_plot = np.array(h5file['t'])
            v = np.array(h5file['v'])
            r_plot = np.array(h5file['r'])
            p_types_raw = h5file['p_types']
            p_types = p_types_raw.asstr()[:]

        if plot_idxs is None:
            n_plots = len(p_types)
            plot_idxs = np.arange(n_plots)
        else:
            n_plots = len(plot_idxs)

        if smoothing:
            r_plot = gaussian_filter1d(r_plot, sigma=sigma)

        fig = plt.figure(figsize=(10, 4.25*n_plots))
        for i_plot, plot_idx in enumerate(plot_idxs):
            plot_loc_1 = int(2 * i_plot + 1)
            plot_loc_2 = int(2 * i_plot + 2)
            ax = fig.add_subplot(n_plots, 2, plot_loc_1)
            v_min, v_max = v.min(), v.max()
            v_mesh = np.linspace(v_min, v_max, bins+1)
            if hide_refractory:
                # take out all values of v[:, k] where v = V_reset
                v_hist = np.array([np.histogram(v[np.where(v[:, k] != self.V_reset), k], bins=v_mesh)[0] for k in range(v.shape[1])])
            else:
                v_hist = np.array([np.histogram(v[:, k], bins=v_mesh)[0] for k in range(v.shape[1])])


            if cutoff == None:
                cutoff = v_hist.max()
            z_max, z_min = cutoff, 0#v_hist.max(), v_hist.min()

            X, Y = np.meshgrid(t_plot, v_mesh[1:])
            c = ax.pcolormesh(X, Y, v_hist.T, cmap='viridis', vmin=z_min, vmax=z_max)
            fig.colorbar(c, ax=ax)

            ax.set_title(f"Membrane potential histogram ({str(p_types[plot_idx])})")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("membrane potential (mv)")
            # ax.set_ylim([self.V_reset * 1.1, self.V_th*1.1])

            ax = fig.add_subplot(n_plots, 2, plot_loc_2)
            ax.plot(t_plot, r_plot * 1000)
            ax.set_title(f"Population activity ({str(p_types[plot_idx])})")
            ax.set_ylabel("Firing rate (Hz)")
            ax.set_xlabel("time (ms)")
            ax.grid()
        plt.tight_layout()
        plt.show()