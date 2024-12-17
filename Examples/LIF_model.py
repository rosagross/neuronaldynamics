from Model.Models import *
import numpy as np


def gen_spikes_func(t_spike=[1.0], spike_value=1.0, step_size=0.01):

    def spikes(t, t_spike=t_spike, spike_value=spike_value, step_size=step_size):
        out = 0
        for i, t_i in enumerate(t_spike):

            if (t <= t_i < t + step_size):
                out += spike_value

        return out

    return spikes

input_func = gen_spikes_func(t_spike=[4.0, 5.0, 6.0], spike_value=1.0e2, step_size=0.01)
system = General1DSystem(model='-x', input_func=input_func)
x0 = np.array([1.0])
t = np.arange(0, 10, 0.01)
system.solve(x0=x0, t=t, tcrit=np.array([4.0, 5.0, 6.0]))
system.plot_solution()

def block_input(t, height, t1, t2):

    if t1<t<t2:
        return height

# TODO: resetting he voltage poses a problem to the formulation of the ODE for now
LIF = General1DSystem(model='-x', input_func=block_input)
LIF.solve(x0=x0, t=t)
LIF.plot_solution()


