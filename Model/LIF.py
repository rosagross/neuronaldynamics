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
from tqdm import tqdm

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
        self.pars['V_reset'] = -75.  # reset potential [mV]
        self.pars['tau_m'] = 10.  # membrane time constant [ms]
        self.pars['g_L'] = 10.  # leak conductance [nS]
        self.pars['V_init'] = -75.  # initial potential [mV]
        self.pars['E_L'] = -75.  # leak reversal potential [mV]
        self.pars['tref'] = 2.  # refractory time (ms)

        # simulation parameters #
        self.pars['T'] = 400.  # Total duration of simulation [ms]
        self.pars['dt'] = .1  # Simulation time step [ms]
        self.pars['weights'] = None


        self.pars['tau_alpha'] = 1/3 # parameters from Nykamp 2000 here
        self.pars['n_alpha'] = 9
        self.pars['n_neurons'] = 1

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
        self.weights = self.pars['weights']
        self.n_neurons = self.pars['n_neurons']
        self.tau_alpha = self.pars['tau_alpha']
        self.n_alpha = self.pars['n_alpha']


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
        if self.Iinj is not None:
          Iinj = self.Iinj

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


            if self.n_neurons > 1:
                for i in range(self.n_neurons):
                    # get input from other neurons
                        if it > 0:
                              input[i, :] = np.convolve(np.sum(self.weights[:, i] * t_spikes.T, axis=1), self.alpha)
                              # connectivity weight is parameter that scales the current here
                              # input = np.convolve(np.sum(self.weights[:, i] * t_spikes.T, axis=1), self.alpha)

                Iin[:, it] = Iinj[it] + input[:, it]
            else:
                Iin = Iinj

            # Calculate the increment of the membrane potential
            dv = (-(self.v[:, it] - self.E_L) + Iin[:, it] / self.g_L) * (self.dt / self.tau_m)

            # Update the membrane potential
            self.v[:, it + 1] = self.v[:, it] + dv

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

    def gen_poisson_spikes_input(self, i_max=300, rate=1, t_start=0.0, t_end=None):
        """
        Generate spike times and currents for a neuron with a time-dependent firing rate using an inhomogeneous Poisson
         process.
        modified from:
        https://medium.com/@baxterbarlow/poisson-spike-generators-stochastic-theory-to-python-code-a76f8cc7cc32

        Parameters:
        imax (float): Max value if I which is used as scaling factor for random sampling
        rate (float): Firing rate at time t (spikes per second).

        """
        if t_end is None:
            t_end = self.T

        t_last_spike = 0
        ts = np.arange(0, self.T, self.dt)
        i_s = np.zeros_like(ts)

        for i, t_i in enumerate(ts):

            if i == 0:
                # TODO:
                #  is need to be gamma distributed according to paper
                interval = -np.log(np.random.rand()) / rate

            if t_i - t_last_spike > interval:
                i_s[i] = np.random.rand() * i_max
                t_last_spike = t_i
                interval = -np.log(np.random.rand()) / rate

        i_s[:int(t_start / self.dt)] = 0
        i_s[int(t_end / self.dt):] = 0
        self.Iinj = i_s
