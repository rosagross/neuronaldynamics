import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import random
from tqdm import tqdm
from Model.LIF import Conductance_LIF
from Model.Nykamp_Model import Nykamp_Model_1
from Utils import compare_firing_rate
matplotlib.use('TkAgg')

########################################################################################################################
# LIF model #
########################################################################################################################

# set-up time and input model
def v0(t):
    v0_bar = 700  # 700 spikes / second / 1000 to convert to ms
    f = 10
    return (v0_bar/1000) * (1 + np.sin(2*np.pi*f/1000*t))

T = 15
dt_LIF = 0.1
t_LIF = np.arange(0.0, T, dt_LIF)
dim = 1000

def sine(t):
    f = 10
    i_ext_0 = 1e-1  # 200µA / 10 mS  =  20mV input
    return i_ext_0 * (1 + np.sin(2*np.pi*f/1000*t))


def step(t):
    t1 = 0
    t2 = 15
    i_0 = 0.6
    res = np.zeros_like(t)
    res[t > t1] = i_0
    res[t > t2] = 0
    return res

i_ext_vals = step(t_LIF)
# i_ext_vals = step(t)
i_ext_vals = i_ext_vals.repeat(dim).reshape(t_LIF.shape[0], dim).T

neuron_parameters = {'T': T, 'tau_m': 20, 't_ref': 0, 'n_neurons': dim, 'Iinj': i_ext_vals, 'init_pdf_sigma':1,
                     'noise_sigma': 0.4, 'V_reset': -70}
lif = Conductance_LIF(parameters=neuron_parameters)

lif.run()

# visualize
# lif.plot_volt_trace(idx=3, population_idx=2)
# lif.raster_plot()
# lif.plot_populations(bins=1000, smoothing=True, sigma=10, hide_refractory=True, cutoff=10, size=1.0, ylims=(-70, -55))

t_NMM = np.arange(0, T, 0.01)
input_current = step(t_NMM)
nmm_parameters = {'T': T, 'dt': 0.01, 'dv': 0.01, 'tau_mem': [20], 'tau_ref': [0], 'current_sigma': 8,
                  'input_function': input_current, 'input_type': 'stochastic-current', 'solver': 'hu-2021',
                  'current_factor': 1.7e-1, 'static_noise': True, 'connectivity_matrix': np.array([[0]]),
                  'init_pdf_weight': 0, 'init_pdf_sigma': 1, 'init_pdf_offset': 5}

NMM = Nykamp_Model_1(parameters=nmm_parameters, name='Nykamp')
NMM.simulate()
NMM.plot(heat_map=True, z_limit=0.004)
r_LIF_time = np.interp(t_LIF, t_NMM, NMM.r[0])
NMM.r = [r_LIF_time]
NMM.clean()
NMM.t = t_LIF
NMM.save_sim_results(r=NMM.r, rho=np.array([0]), rho_plot=np.array([0]))

compare_firing_rate('Nykamp', 'Conductance_LIF')
Conductance_LIF.clean()
NMM.clean()




