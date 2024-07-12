from NeuronModels import *
import numpy as np

class1 = Test()
class1.test_function('test1')

system = general_2D_system()
x0 = np.array([1., 0.2])
t = np.arange(0, 10, 0.01)
system.solve(x0=x0, t=t)
# system.plot_solution()

system_1 = general_2D_system(model=['y', '-a*x - b*y'], model_name='damped harmonic oscillator',
                             parameters={'a': 0.2, 'b': 1.0})
x0 = np.array([1., 0.2])
t = np.arange(0, 20, 0.01)
t_phase = np.arange(0, 3, 0.1)
system_1.solve(x0=x0, t=t)
system_1.plot_solution()

# system_1.nullclines = ['-b/a*x', '0']
system_1.plot_phase(t=t_phase, plot_nullclines=True)

