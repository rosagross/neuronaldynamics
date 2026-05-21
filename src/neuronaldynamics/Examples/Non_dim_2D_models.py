from Model.Models import *
import numpy as np

system = General2DSystem(model=['a + b*x**2', '-y'], model_name='SN-test',
                           parameters={'a': -1.5, 'b': 1.0}, usetex=False)
x0 = np.array([1., 0.2])
t = np.arange(0, 20, 0.01)
t_phase = np.arange(0, 3, 0.1)
system.solve(x0=x0, t=t)
system.plot_solution()
system.solve(x0=[-2, -2], t=t)
system.plot_solution()
system.get_equilibria(x_range=[-4, 10], y_range=[-4, 10], x_steps=2, y_steps=2)
system.phase_portrait(x_range=[-4, 10], y_range=[-4, 10], x_steps=0.5, y_steps=0.5)

