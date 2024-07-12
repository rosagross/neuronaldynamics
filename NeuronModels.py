from Solvers import Euler
import warnings
import numpy as np
import sympy as sy
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class Test():
    """ Test function for testing general coding
    This function will be deleted once its tests have passed"""

    def __init__(self):
        self.status = 'init'

    def test_function(self, message):
        print(message)
        print(f'status is {self.status}')

class general_2D_system():
    """ A class to investigate general 2D systems in terms of their dynamics """

    def __init__(self, model_name=None,  model=None, variables=None, parameters=None, t=None, solver=None):
        self.model_name = '2D_system'
        self.model = ['y', '-x']
        self.variables = ['x', 'y']
        self.parameters = {}
        self.t = np.arange(0, 10, 0.01)
        self.solver = 'odeint'
        self.time_vals = None

        self.nullclines=[]

        if model_name != None:
            self.model_name = model_name
        if model != None:
            self.model = model
        if variables != None:
            self.variables = variables
        if parameters != None:
            self.parameters = parameters
        if solver != None:
            self.solver = solver


    def system(self, u, t):
        out = np.zeros_like(u)
        for _, pname in enumerate(self.parameters.keys()):
            exec(f"{pname} = self.parameters['{pname}']")
        exec(f'{self.variables[0]}, {self.variables[1]} = u')
        out[0] = eval(self.model[0])
        out[1] = eval(self.model[1])
        return out

    def solve(self, x0, t, step_size=0.1, **kwargs):
        if self.solver == 'odeint':
            self.sol = odeint(self.system, x0, t, **kwargs)
            self.time_vals = t

    def plot_solution(self, save_fig=False, fig_fname='test.png'):
        if type(self.time_vals) != np.ndarray:
            raise AttributeError("Solution hasn't been evaluated yet!")

        plt.plot(self.time_vals, self.sol[:, 0], label=self.variables[0])
        plt.plot(self.time_vals, self.sol[:, 1], label=self.variables[1])
        plt.legend(loc='best')
        plt.title(self.model_name)
        plt.xlabel('t')
        plt.show()
        if save_fig:
            plt.savefig(fig_fname)

    def plot_phase(self, t, x_lim=[-5, 5], y_lim=[-5, 5], x_density=10, y_density=10, x_label=None, y_label=None,
                   save_fig=False, fig_fname='2D_system_phase_space.png', plot_nullclines=True):

        if x_label == None:
            x_label = self.variables[0]
        if y_label == None:
            y_label = self.variables[1]

        x = np.linspace(x_lim[0], x_lim[1], x_density)
        y = np.linspace(y_lim[0], y_lim[1], y_density)
        vect0 = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

        fig = plt.figure()

        #####################################
        # generate the phase space streamplot
        #####################################

        for i in range(vect0.shape[0]):
            x0 = [vect0[i][0], vect0[i][1]]
            self.solve(x0=x0, t=t)
            sol = self.sol
            plt.quiver(sol[:-1, 0], sol[:-1, 1], sol[1:, 0] - sol[:-1, 0], sol[1:, 1] - sol[:-1, 1], scale_units='xy',
                       angles='xy', scale=2, color='k', alpha=0.5)
            plt.plot(sol[:, 0], sol[:, 1], color='k', alpha=0.4)

        #########################################
        # optionally add nullclines and equilibira
        #########################################
        if plot_nullclines:
            if len(self.nullclines) == 0:

                self.get_nullclines()
                print(f'found nullclines: {self.nullclines}')

            x_length = x_lim[1] - x_lim[0]
            xs = np.arange(x_lim[0], x_lim[1], x_length/1000)
            for _, pname in enumerate(self.parameters.keys()):
                exec(f"{pname} = self.parameters['{pname}']")
            nullcline_dict = self.parameters.copy()
            nullcline_dict[f'{self.variables[0]}'] = xs
            x_nullcline_values = eval(f'{self.nullclines[0]}', nullcline_dict)
            y_nullcline_values = eval(f'{self.nullclines[1]}', nullcline_dict)

            # avoid dimensionality error if nullcline is constant
            # if y nullcline is constant plot a vline at x=constant
            if type(x_nullcline_values) != np.ndarray:
                x_nullcline_values = x_nullcline_values * np.ones_like(xs)
            if type(y_nullcline_values) != np.ndarray:
                plt.vlines(x=y_nullcline_values, ymin=y_lim[0], ymax=y_lim[1], colors='b', label=y_label + ' nullcline')
            else:
                plt.plot(xs, y_nullcline_values, 'b', label=y_label + ' nullcline')

            plt.plot(xs, x_nullcline_values, 'orange', label=x_label + ' nullcline')
            plt.legend(loc='best')

        plt.xlim(0.95*x_lim[0], 0.95*x_lim[1])
        plt.ylim(0.95*y_lim[0], 0.95*y_lim[1])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(self.model_name + 'phase_space')
        plt.show()
        if save_fig:
            plt.savefig(fig_fname)

    def get_nullclines(self):

        sympy_var_1, sympy_var_2 = sy.symbols(f'{self.variables[0]}, {self.variables[1]}')

        x_expr = sy.parsing.sympy_parser.parse_expr(self.model[0], evaluate=False)
        x_nullcline = sy.solve(x_expr, sympy_var_2)
        if x_nullcline == []:
            x_nullcline = sy.solve(x_expr, sympy_var_1)
        if len(x_nullcline) > 1:
            warnings.warn(f'x_nullcline has multiple solutions for y: {x_nullcline}')
        y_expr = sy.parsing.sympy_parser.parse_expr(self.model[1], evaluate=False)
        y_nullcline = sy.solve(y_expr, sympy_var_2)
        self.nullclines = [str(x_nullcline[0]), str(y_nullcline[0])]