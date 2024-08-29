from Model.Damped_harmonic_oscillator import Damped_harmonic_oscillator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np

#########################################################
#                       1D Example                      #
#########################################################

parameters = {'a': 1.1, 'b': 1.0}
system = Damped_harmonic_oscillator(parameters=parameters)

x0 = np.array([1., 0.2])
t = np.arange(0, 20, 0.01)
system.solve(x0=x0, t=t)
system.plot_solution()

test_solution = system.sol[:, 0]
system.parameters['a'] = 0.5
system.parameter_fit(target_series=test_solution, parameter_bounds=[0.0, 2.0], x0=x0, t=t, eps=1e-4,
                     verbose=1)

test_solution = test_solution[:, np.newaxis]
system.plot_solution(x_compare=test_solution, compare_idx=0, title=f'parameter fit a={system.parameters["a"]:.5f}'
                                                                   f' with nrmsd = {system.fit_error*100:.5f}\%')



#########################################################
#                       2D Example                      #
#########################################################

# get solution for original value
parameters = {'a': 1.1, 'b': 1.0}
system = Damped_harmonic_oscillator(parameters=parameters)
x0 = np.array([1., 0.2])
t = np.arange(0, 20, 0.01)
system.solve(x0=x0, t=t)
test_solution_2D = system.sol

# guess parameter a
system.parameters['a'] = 0.5
# run fit for 2D
system.parameter_fit(target_series=test_solution_2D, variables=['x', 'y'], parameter_bounds=[0.0, 2.0], x0=x0, t=t, eps=1e-4,
                     verbose=1)
system.plot_solution(x_compare=test_solution_2D, compare_idx=[0, 1], title=f'parameter fit a={system.parameters["a"]:.5f}'
                                                                   f' with nrmsd = {system.fit_error*100:.5f}\%')