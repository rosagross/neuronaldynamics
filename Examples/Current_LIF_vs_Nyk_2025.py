import numpy as np
import matplotlib
import random
from tqdm import tqdm
from Model.LIF import Conductance_LIF
from Model.Nykamp_Model import Nykamp_Model_1
from Utils import compare_firing_rate
matplotlib.use('TkAgg')

########################################################################################################################
# LIF model #
########################################################################################################################

T = 150
dt = 0.1
t = np.arange(0.0, T, dt)

# set up model
dim = 1000
con = np.zeros((dim, dim))
n_connection = 0  # int(dim/10)

for i in tqdm(range(dim), f'computing random neuron connections for {dim} neurons'):
    possible_connections = np.arange(dim)
    # exclude current idx
    possible_connections = possible_connections[possible_connections!=i]
    connections_i = random.sample(possible_connections.tolist(), n_connection)
    if np.unique(np.array(connections_i)).shape[0] != n_connection:
        print(f"failed sampling!!!, idx {i}")
    con[i, connections_i] = 1



def step_µA(t):
    t1 = 20
    t2 = 90
    i_0 = 1
    res = np.zeros_like(t)
    res[t > t1] = i_0
    res[t > t2] = 0
    return res

def step_A(t):
    t1 = 20
    t2 = 90
    i_0 = 1e6
    res = np.zeros_like(t)
    res[t > t1] = i_0
    res[t > t2] = 0
    return res


i_ext_vals = step_µA(t)
i_ext_vals = i_ext_vals.repeat(dim).reshape(t.shape[0], dim).T

neuron_parameters = {'T': T, 'tau_m': 20, 't_ref': 3, 'weights': con, 'n_neurons': dim, 'Iinj': i_ext_vals}
lif = Conductance_LIF(parameters=neuron_parameters)

lif.run()

# visualize
# lif.plot_volt_trace(idx=3, population_idx=2)
# lif.raster_plot()
lif.plot_populations(bins=1000, smoothing=True, sigma=10, hide_refractory=True, cutoff=None, size=1.0)

########################################################################################################################
# Nykamp model #
########################################################################################################################


pars_1D = {}
pars_1D['connectivity_matrix'] = np.array([[0]])
pars_1D['u_rest'] = -65
pars_1D['u_thr'] = -55
pars_1D['u_exc'] = 0
pars_1D['u_inh'] = -70
pars_1D['tau_mem'] = np.array([20])
pars_1D['tau_ref'] = np.array([3])

# pars_1D['input_function'] = step_population
pars_1D['input_function'] = step_A
pars_1D['input_function_type'] = 'custom'
pars_1D['input_function_idx'] = [0, 0]
pars_1D['population_type'] = ['exc']
pars_1D['input_type'] = 'current-2'
pars_1D['c_mem'] = [0.2]  # 0.2F capacitance
pars_1D['T'] = T

nyk1D = Nykamp_Model_1(parameters=pars_1D, name='Nykamp')
nyk1D.simulate()
nyk1D.plot(heat_map=True, z_limit=0.2)
#
compare_firing_rate('Nykamp', 'Conductance_LIF')


