from Model.Models import *
import numpy as np


# test parametrized WC model without excitation
Neuron_model_1 = ['(1/tau)*(-r_e + k - q*r_e)*(1+np.exp(-(c_ee*r_e + c_ei*r_i)))**(-1)',
                  '(1/tau)*(-r_i + k - q*r_i)*(1+np.exp(-(c_ie*r_e + c_ii*r_i)))**(-1)']
system = General2DSystem(model=Neuron_model_1, model_name='Wilson-Cowan',
                           variables=['r_e', 'r_i'], parameters={'tau': 10.0, 'k': 1.0, 'q': 2.0, 'c_ee': 15,
                                                                 'c_ei': 15, 'c_ie': -15, 'c_ii': -4})
t = np.arange(0, 100, 0.05)
x0 = np.array([1.0, 0.5])
system.solve(x0=x0, t=t)
# system.plot_solution()
#TODO: nullclines are independent on variables, straight lines, plotting them freezes matplotlib, find out why
system.plot_phase(t=t, plot_nullclines=True, get_equilibiria=True, x_lim=[0, 0.5], y_lim=[0, 0.3], x_density=10,
                  y_density=20,  quiver_scale=20)

parameters={'tau': 10.0, 'k': 0.3, 'q': 2.0, 'c_ee': 15, 'c_ei': 15, 'c_ie': -15, 'c_ii': -4}
system1 = General2DSystem(model=Neuron_model_1, model_name='Wilson-Cowan',
                           variables=['r_e', 'r_i'], parameters=parameters)
system.plot_phase(t=t, plot_nullclines=False, get_equilibiria=False, x_lim=[0, 0.5], y_lim=[0, 0.3], x_density=10,
                  y_density=20,  quiver_scale=20)
