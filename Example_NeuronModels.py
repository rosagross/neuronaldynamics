from NeuronModels import *
import numpy as np

#TODO: turn this into a jupyter Notebook
#TODO: apply range of step currents to test usual behaviour

#TODO: create sympy function from matrix
# fix error 'TypeError: can't convert expression to float' line 208, in get_nullclines_and_jacobian

class1 = Test()
class1.test_function('test1')

system = general_2D_system()
x0 = np.array([1., 0.2])
t = np.arange(0, 10, 0.01)
system.solve(x0=x0, t=t)
# system.plot_solution()

system_1 = general_2D_system(model=['y', '-a*x - b*y'], model_name='damped harmonic oscillator',
                             parameters={'a': 1.2, 'b': 1.0})
x0 = np.array([1., 0.2])
t = np.arange(0, 20, 0.01)
t_phase = np.arange(0, 3, 0.1)
system_1.solve(x0=x0, t=t)
# system_1.plot_solution()

# system_1.nullclines = ['-b/a*x', '0']
system_1.plot_phase(t=t_phase, plot_nullclines=True)

# try out example for I_{Na, p} + I_K model

Neuron_model_1 = ['-5*(V-2)**3 + (V-2)**2 + 5*(V-2) + 2 -n', '0.015*(V-0.5)**8-n +0.5']
plt.rcParams['text.usetex'] = True
system_2 = general_2D_system(model=Neuron_model_1, model_name=r'I\_{Na, p} + I_K \textit{model approx.}',
                             variables=['V', 'n'])
t = np.arange(0, 10, 0.01)
x0 = np.array([0, 0])
system_2.solve(x0=x0, t=t)
# system_2.plot_solution()
system_2.plot_phase(t=t, plot_nullclines=True, x_lim=[0, 3], y_lim=[0, 5], x_density=20, y_density=10,
                    quiver_scale=1)
