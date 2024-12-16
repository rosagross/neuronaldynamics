from Model.Models import *
import numpy as np

#TODO: turn this into a jupyter Notebook
#TODO: apply range of step currents to test usual behaviour

def gen_spikes_func(t_spike=[1.0], spike_value=1.0, step_size=0.01):

    def spikes(t, t_spike=t_spike, spike_value=spike_value, step_size=step_size):

        for _, t_i in enumerate(t_spike):
            # if (t < t_i < t + step_size):
            #     return spike_value
            # else:
            #     return 0
            counter = 0
            if t > t_i and counter == 0:
                counter = 1
                return spike_value
            else:
                return 0

    return spikes

input_func = gen_spikes_func(t_spike=[6.0], spike_value=1.0, step_size=0.01)
system = General1DSystem(model='-x', input_func=input_func)
x0 = np.array([1.0])
t = np.arange(0, 10, 0.01)
system.solve(x0=x0, t=t)
system.plot_solution()


