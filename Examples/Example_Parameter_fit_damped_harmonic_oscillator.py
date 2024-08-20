from Model.Damped_harmonic_oscillator import Damped_harmonic_oscillator
import numpy as np

parameters = {'a': 1.1, 'b': 1.0}
system = Damped_harmonic_oscillator(parameters=parameters)

x0 = np.array([1., 0.2])
t = np.arange(0, 20, 0.01)
t_phase = np.arange(0, 3, 0.1)
system.solve(x0=x0, t=t)
system.plot_solution()

test_solution = system.sol[:, 0]
system.parameters['a'] = 0.5
system.parameter_fit(target_series=test_solution, parameter_bounds=[0.0, 2.0], x0=x0,
                     verbose=True)

system.plot_solution(x_compare=test_solution, compare_idx=0)
