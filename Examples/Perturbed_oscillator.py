from Model.Models import *
import numpy as np

system = General2DSystem(model=['y', '-(k/m)*x'], model_name='harmonic oscillator',
                           parameters={'k': 1, 'm': 1}, usetex=False)
x0 = np.array([0.1, 0.0])
t = np.arange(0, 20, 0.01)
# system.solve(x0=x0, t=t)
# system.plot_solution()
# system.solve(x0=[-2, -2], t=t)
# system.plot_solution()
# system.get_equilibria(x_range=[-5, 5], y_range=[-5, 5], x_steps=2, y_steps=2)
# system.phase_portrait(x_range=[-5, 5], y_range=[-5, 5], x_steps=0.5, y_steps=0.5)

# u = lambda x: np.array([1.0, 0.0])
# system2 = General2DSystem(model=['y', '-a*y-(k/m)*x'], model_name='harmonic oscillator',
#                            parameters={'a': 0.00, 'k': 1, 'm': 1}, input_func=u, usetex=False)
# system2.solve(x0=[0.1, 0.0], t=t)
# system2.plot_solution()
# system2.get_equilibria(x_range=[-5, 5], y_range=[-5, 5], x_steps=2, y_steps=2)
# system2.phase_portrait(x_range=[-5, 5], y_range=[-5, 5], x_steps=0.5, y_steps=0.5, plot_solution=1)

#TODO: plot periodic perturbation, plot damped with forcing, look at case of limit cycle with unstable equilibrium inside
# when can oscillation be destroyed? (stable LC with noise background)

# u = lambda x: np.array([0.5* np.sin((1/20)*x), 0.0])
# system3 = General2DSystem(model=['y', '-a*y-(k/m)*x'], model_name='harmonic oscillator',
#                            parameters={'a': 0.00, 'k': 1, 'm': 1}, input_func=u, usetex=False)
# t = np.arange(0.0, 200, 0.1)
# system3.solve(x0=[1.0, 0.0], t=t)
# system3.plot_solution()
# system3.get_equilibria()
# system3.phase_portrait(x_range=[-5, 5], y_range=[-5, 5], x_steps=0.5, y_steps=0.5, plot_solution=1)

# u = lambda x: np.array([0.5* np.sin(1.5*x) + 0.5*np.sin(0.5*x), 0.0])
# system4 = General2DSystem(model=['y', '-a*y-(k/m)*x'], model_name='harmonic oscillator',
#                            parameters={'a': 0.00, 'k': 1, 'm': 1}, input_func=u, usetex=False)
# t = np.arange(0.0, 50, 0.1)
# system4.solve(x0=[1.0, 0.0], t=t)
# system4.plot_solution()
# system4.get_equilibria()
# system4.phase_portrait(x_range=[-5, 5], y_range=[-5, 5], x_steps=0.5, y_steps=0.5, plot_solution=1)



def pulse(t, amp=0.5, start=5, end=10):
    if t > start and t < end:
        return np.array([amp, 0])
    else:
        return np.array([0, 0])

def rect(t, amp=0.5, modval=1, period=2*np.pi):
    if t % period < modval:
        return np.array([amp, 0])
    else:
        return np.array([0, 0])
u = rect
system5 = General2DSystem(model=['y', '-a*y-(k/m)*x'], model_name='harmonic oscillator',
                           parameters={'a': 0.00, 'k': 1, 'm': 1}, input_func=u, usetex=False)
t = np.arange(0.0, 50, 0.1)
system5.solve(x0=[1.0, 0.0], t=t)
system5.plot_solution()
system5.get_equilibria()
system5.phase_portrait(x_range=[-5, 5], y_range=[-5, 5], x_steps=0.5, y_steps=0.5, plot_solution=1)

