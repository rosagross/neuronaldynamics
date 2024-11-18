"""
Code from Richard Naud:
https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html
Modified by Erik Müller

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.special import factorial
from scipy.stats import gamma
import random
from tqdm import tqdm
from Utils import raster, nrmse
import time

class LIF_population():

    def __init__(self, **kwargs):
        # Set parameters
        self.default_pars(**kwargs)
        self.Iinj = None


    def default_pars(self, **kwargs):
        """
        Function that sets default parameters
        :param kwargs:
        :return:
        """
        self.pars = {}

        # typical neuron parameters#
        self.pars['V_th'] = -55.  # spike threshold [mV]
        self.pars['V_reset'] = -65.  # reset potential [mV]
        self.pars['tau_m'] = 10.  # membrane time constant [ms]
        self.pars['g_L'] = 10.  # leak conductance [nS]
        self.pars['V_init'] = -65.  # initial potential [mV]
        self.pars['E_L'] = -65.  # leak reversal potential [mV]
        self.pars['tref'] = 2.  # refractory time (ms)

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

    def run(self, Iinj=None, stop=False, custom_i=True):
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
                self.Iext = np.repeat(self.Iext, self.n_neurons)
                self.Iext = np.reshape(self.Iext, (self.n_neurons, Iext_shape_init))

        self.get_alpha_kernel()

        # Initialize voltage
        self.v = np.zeros((self.n_neurons, self.Lt))
        t_spikes = np.zeros_like(self.v)
        Iin = np.zeros_like(self.v)
        self.v[:, 0] = self.V_init
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

        conv_dummy = np.convolve(self.v[0], self.alpha)
        conv_shape = conv_dummy.shape
        input = np.zeros((self.n_neurons, conv_shape[0]))

        # record spike times
        for _ in range(self.n_neurons):
            self.rec_spikes.append([])

        if self.verbose > 0:
            t2 = time.time()
            print(f'set-up: {t2-t1:.4f}s')

        for it in tqdm(range(self.Lt - 1)):
            for i in range(self.n_neurons):

                if tr[i] > 0:  # check if in refractory period
                    self.v[i, it] = self.V_reset  # set voltage to reset
                    tr[i] = tr[i] - 1  # reduce running counter of refractory period

                elif self.v[i, it] >= self.V_th:  # if voltage over threshold
                    self.rec_spikes[i].append(it)  # record spike event
                    t_spikes[i, it] = 1
                    self.r[i, int(t_last_spike[i]):it] = 1000 / (
                          it - t_last_spike[i]) * self.dt  # times 1000 for conversion 1/ms -> Hz
                    t_last_spike[i] = it
                    self.v[i, it] = self.V_reset  # reset voltage
                    tr[i] = self.tref / self.dt  # set refractory time

            if self.verbose > 0:
                t_conv_1 = time.time()
            if self.n_neurons > 1:
                # problem with this version: keeps the dimension fixed while np.convolve extends for overshoot
                # test = np.einsum('i,kl->kl', self.alpha, np.sum(self.weights[:, i] * np.reshape(np.repeat(t_spikes, 50), (5000, 50, 50)), axis=1)).T
                for i in range(self.n_neurons):
                    # get input from other neurons
                        if it > 0:
                            # TODO: This is a bottleneck-find out if this can be done faster maybe through vectorization
                              input[i, :] = np.convolve(np.sum(self.weights[:, i] * t_spikes.T, axis=1), self.alpha)
                              # connectivity weight is parameter that scales the current here
                              # input = np.convolve(np.sum(self.weights[:, i] * t_spikes.T, axis=1), self.alpha)

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
      plt.tight_layout()
      plt.show()

    def gen_poisson_spikes_input(self, i_max=300, rate=1, mu=0.008, coeff_of_var=0.5, t_start=0.0, t_end=None):
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

        if t_end is None:
            t_end = self.T


        ts = np.arange(0, self.T, self.dt)
        i_s = np.zeros((self.n_neurons, ts.shape[0]))
        for j in range(self.n_neurons):
            t_last_spike = 0
            for i, t_i in enumerate(ts):

                if i == 0:
                    # TODO:
                    #  is need to be gamma distributed according to paper
                    interval = -np.log(np.random.rand()) / rate

                if t_i - t_last_spike > interval:
                    sign = [-1,1][random.randrange(2)]
                    i_s[j, i] = gamma_pdf.rvs(size=1) * i_max * sign
                    t_last_spike = t_i
                    interval = -np.log(np.random.rand()) / rate

        i_s[:int(t_start / self.dt)] = 0
        i_s[int(t_end / self.dt):] = 0
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

    def plot_firing_rate(self, neuron_num):
        fig = plt.figure(figsize=(8, 8))
        # TODO: find out how to make this plot the same way it is in the paper
        # also create a spike raster plot and check for availabilty of this via pyrates
        for n, n_neuron in enumerate(neuron_num):
            ax = fig.add_subplot(len(neuron_num), 1, int(n + 1))
            ax.hist(self.r[n_neuron, :], bins=100, density=True, alpha=0.7)
            ax.plot(np.mean(self.r, axis=0))
            ax.set_ylabel('r in Hz')
        plt.tight_layout()
        ax.set_xlabel('time in ms')
        plt.show()

    def plot_voltage_hist(self, times):
        fig = plt.figure(figsize=(8, 8))


        for n, time in enumerate(times):
            ax = fig.add_subplot(len(times), 1, int(n + 1))
            ax.hist(self.v[:, time], bins=100, density=True, alpha=0.7)

        plt.tight_layout()
        ax.set_xlabel('V in mv')
        plt.show()