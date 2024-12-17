"""
Code from Richard Naud:
https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html
Modified by Erik Müller

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from tqdm import tqdm
matplotlib.use('TkAgg')


def default_pars(**kwargs):
  pars = {}

  # typical neuron parameters#
  pars['V_th'] = -55.     # spike threshold [mV]
  pars['V_reset'] = -75.  # reset potential [mV]
  pars['tau_m'] = 10.     # membrane time constant [ms]
  pars['g_L'] = 10.       # leak conductance [nS]
  pars['V_init'] = -75.   # initial potential [mV]
  pars['E_L'] = -75.      # leak reversal potential [mV]
  pars['tref'] = 2.       # refractory time (ms)

  # simulation parameters #
  pars['T'] = 400.  # Total duration of simulation [ms]
  pars['dt'] = .1   # Simulation time step [ms]

  # external parameters if any #
  for k in kwargs:
    pars[k] = kwargs[k]

  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # Vector of discretized time points [ms]

  return pars


def plot_volt_trace(pars, v, sp):
  """
  Plot trajetory of membrane potential for a single neuron

  Expects:
  pars   : parameter dictionary
  v      : volt trajetory
  sp     : spike train

  Returns:
  figure of the membrane potential trajetory for a single neuron
  """

  V_th = pars['V_th']
  dt, range_t = pars['dt'], pars['range_t']
  if sp.size:
    sp_num = (sp / dt).astype(int) - 1
    v[sp_num] += 20  # draw nicer spikes

  plt.plot(pars['range_t'], v, 'b')
  plt.axhline(V_th, 0, 1, color='k', ls='--')
  plt.xlabel('Time (ms)')
  plt.ylabel('V (mV)')
  plt.legend(['Membrane\npotential', r'Threshold V$_{\mathrm{th}}$'],
             loc=[1.05, 0.75])
  plt.ylim([-80, -40])
  plt.tight_layout()
  plt.show()


def plot_GWN(pars, I_GWN):
  """
  Args:
    pars  : parameter dictionary
    I_GWN : Gaussian white noise input

  Returns:
    figure of the gaussian white noise input
  """

  plt.figure(figsize=(12, 4))
  plt.subplot(121)
  plt.plot(pars['range_t'][::3], I_GWN[::3], 'b')
  plt.xlabel('Time (ms)')
  plt.ylabel(r'$I_{GWN}$ (pA)')
  plt.subplot(122)
  plot_volt_trace(pars, v, sp)
  plt.tight_layout()
  plt.show()


def my_hists(isi1, isi2, cv1, cv2, sigma1, sigma2):
  """
  Args:
    isi1 : vector with inter-spike intervals
    isi2 : vector with inter-spike intervals
    cv1  : coefficient of variation for isi1
    cv2  : coefficient of variation for isi2

  Returns:
    figure with two histograms, isi1, isi2

  """
  plt.figure(figsize=(11, 4))
  my_bins = np.linspace(10, 30, 20)
  plt.subplot(121)
  plt.hist(isi1, bins=my_bins, color='b', alpha=0.5)
  plt.xlabel('ISI (ms)')
  plt.ylabel('count')
  plt.title(r'$\sigma_{GWN}=$%.1f, CV$_{\mathrm{isi}}$=%.3f' % (sigma1, cv1))

  plt.subplot(122)
  plt.hist(isi2, bins=my_bins, color='b', alpha=0.5)
  plt.xlabel('ISI (ms)')
  plt.ylabel('count')
  plt.title(r'$\sigma_{GWN}=$%.1f, CV$_{\mathrm{isi}}$=%.3f' % (sigma2, cv2))
  plt.tight_layout()
  plt.show()


def gen_poisson_spikes(T, dt=0.001, i_max=300, rate=1):
  """
  Generate spike times and currents for a neuron with a time-dependent firing rate using an inhomogeneous Poisson
   process.
  modified from:
  https://medium.com/@baxterbarlow/poisson-spike-generators-stochastic-theory-to-python-code-a76f8cc7cc32

  Parameters:
  rate (float): Firing rate at time t (spikes per second).
  T (float): Total duration of the simulation (seconds).
  dt (float): Time step for simulation (seconds).
  imax (float): Max value if I which is used as scaling factor for random sampling

  Returns:
  spike_times (list): List of spike times.
  """
  t_last_spike = 0
  ts = np.arange(0, T, dt)
  i_s = np.zeros_like(ts)

  for i, t_i in enumerate(ts):

    if i == 0:
      interval = -np.log(np.random.rand()) / rate

    if t_i - t_last_spike > interval:
      i_s[i] = np.random.rand() * i_max
      t_last_spike = t_i
      interval = -np.log(np.random.rand()) / rate

  return ts, i_s

def run_LIF(pars, Iinj, stop=False, custom_i=False, n_neurons=2, alpha=None, weights=None):
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

  # Set parameters
  V_th, V_reset = pars['V_th'], pars['V_reset']
  tau_m, g_L = pars['tau_m'], pars['g_L']
  V_init, E_L = pars['V_init'], pars['E_L']
  dt, range_t = pars['dt'], pars['range_t']
  Lt = range_t.size
  tref = pars['tref']

  # Initialize voltage
  v = np.zeros((n_neurons, Lt))
  t_spikes = np.zeros_like(v)
  Iin = np.zeros_like(Iinj)
  v[:, 0] = V_init
  tr = np.zeros(n_neurons) # the count for refractory duration
  t_last_spike = np.zeros(n_neurons)
  r = np.zeros_like(v)

  if not custom_i:

    # Set current time course
    Iinj = Iinj * np.ones(Lt)

    # If current pulse, set beginning and end to 0
    if stop:
      Iinj[:int(len(Iinj) / 2) - 1000] = 0
      Iinj[int(len(Iinj) / 2) + 1000:] = 0

  # Loop over time
  rec_spikes = []

  # record spike times
  for _ in range(n_neurons):
    rec_spikes.append([])


  for it in tqdm(range(Lt - 1)):
    for i in range(n_neurons):

      # get input from other neurons
      if n_neurons > 1:
        if it > 0:
          # connectivity weight is parameter that scales the current here
          input = np.convolve(np.sum(weights[:, i] * t_spikes.T, axis=1), alpha)
          Iin[it] = Iinj[it] + input[it]

      else:
        Iin = Iinj

      if tr[i] > 0:  # check if in refractory period
        v[i, it] = V_reset  # set voltage to reset
        tr[i] = tr[i] - 1 # reduce running counter of refractory period

      elif v[i, it] >= V_th:  # if voltage over threshold
        rec_spikes[i].append(it)  # record spike event
        t_spikes[i, it] = 1
        r[i, int(t_last_spike[i]):it] = 1000/(it - t_last_spike[i])*dt # times 1000 for conversion 1/ms -> Hz
        t_last_spike[i] = it
        v[i, it] = V_reset  # reset voltage
        tr[i] = tref / dt  # set refractory time

     # Calculate the increment of the membrane potential
      dv = (-(v[i, it] - E_L) + Iin[it] / g_L) * (dt / tau_m)

      # Update the membrane potential
      v[i, it + 1] = v[i, it] + dv

  for i in range(n_neurons):
    # Get spike times in ms
    rec_spikes[i] = np.array(rec_spikes[i]) * dt

  rec_spikes = rec_spikes
  return v, rec_spikes, r



rate = 10
T = 500
t_end = 400
t_start = 0
dt = 0.1
t = np.arange(0.0, T, dt)

t_alpha = t[t < 10]
tau_alpha = 1/3
n_alpha = 9
alpha = np.exp(-t_alpha/tau_alpha) / (tau_alpha * scipy.special.factorial(n_alpha-1)) * (t_alpha/tau_alpha)**(n_alpha-1)
alpha = alpha/np.trapz(alpha, dx=dt)

#TODO:
# is need to be gamma distributed according to paper

# ts, i_s = gen_poisson_spikes(T=T, dt=dt, rate=0.03, i_max=2e4)
i_s = 150*np.ones(int(T/dt))
i_s[:int(t_start/dt)] = 0
i_s[int(t_end/dt):] = 0
w0 = 30
dim = 1
# con = w0*(np.ones((dim, dim)) - np.eye(dim))
con = w0*np.random.uniform(size=(dim, dim))
np.fill_diagonal(con, 0)
# con = np.array([[100, 500], [700, 100]])

# Get parameters
pars = default_pars(T=500, dt=dt)
# Simulate LIF model
v, sp, r = run_LIF(pars, Iinj=i_s, stop=True, custom_i=False, weights=con, alpha=alpha, n_neurons=dim)

# Visualize
plot_volt_trace(pars, v[0], sp[0])
# plot_volt_trace(pars, v[-1], sp[-1])
fig = plt.figure(figsize=(8, 8))
times = [500, 1000, 2000, 3000, 4000]

for n, time in enumerate(times):
  ax = fig.add_subplot(len(times), 1, int(n+1))
  ax.hist(v[:, time], bins=100, density=True, alpha=0.7)

plt.tight_layout()
ax.set_xlabel('V in mv')
# plt.show()

fig = plt.figure(figsize=(8, 8))
neuron_num = [0, 2, 5, 12, 22]

for n, n_neuron in enumerate(neuron_num):
  ax = fig.add_subplot(len(times), 1, int(n+1))
  # ax.hist(r[n_neuron, :], bins=100, density=True, alpha=0.7)
  ax.plot(np.mean(r, axis=0))
  ax.set_ylabel('r in Hz')
plt.tight_layout()
ax.set_xlabel('time in ms')
# plt.show()
# plt.subplots_adjust(hspace=0.5)

# plt.hist(v[:, 3000], bins=100, density=True, alpha=0.7)


print(f'neuron 1 spikes: {sp[0].shape}')
# print(f'neuron 2 spikes: {sp[1].shape}')

