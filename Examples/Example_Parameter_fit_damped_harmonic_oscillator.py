from Model.Damped_harmonic_oscillator import Damped_harmonic_oscillator
import numpy as np

parameters = {'a': 1.1, 'b': 1.0}
system = Damped_harmonic_oscillator(parameters=parameters)

x0 = np.array([1., 0.2])
t = np.arange(0, 20, 0.01)
t_phase = np.arange(0, 3, 0.1)
system.solve(x0=x0, t=t)
system.plot_solution(save_fig=True, fig_fname='osc_test.png')