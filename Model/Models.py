from Solvers.Solvers import Euler
import warnings
import numpy as np
import sympy as sy
from scipy.integrate import odeint
from Utils import get_stability_2D, nrmse
import matplotlib.pyplot as plt
import matplotlib

class General2DSystem():
    """ A class to investigate general 2D systems in terms of their dynamics """

    def __init__(self, model_name=None,  model=None, variables=None, parameters=None, input_func=None,
                 t=None, solver=None, usetex=True):
        self.model_name = '2D_system'
        self.model = ['y', '-x']
        self.variables = ['x', 'y']
        self.parameters = {}
        self.t = np.arange(0, 10, 0.01)
        self.solver = 'odeint'
        self.time_vals = None
        self.nullclines=[]

        if t != None:
            self.t = t
        if model_name != None:
            self.model_name = model_name
        if model != None:
            self.model = model
        if variables != None:
            self.variables = variables

        self.dynamic_parameters = False
        if parameters != None:
            self.parameters = parameters

            # opportunity to init parameters as a ramp in the init
            for i_p, p in enumerate(parameters.keys()):
                if type(self.parameters[p]) != np.array:
                    if self.dynamic_parameters:
                        self.parameters[p] = np.ones_like(self.t)
                else:
                    self.dynamic_parameters = True

        if input_func != None:
            self.input_func = input_func
        else:
            self.input_func = lambda t: np.zeros(2)
        if solver != None:
            self.solver = solver

        self.sol = None

        if usetex:
            plt.rcParams['text.usetex'] = True

        matplotlib.use('TkAgg')


    def system(self, u, t):

        out = np.zeros_like(u)
        # define internal parameters by their names as string
        # this is used to trick python into defining a variable with variable name
        for _, pname in enumerate(self.parameters.keys()):
            if self.dynamic_parameters:
                if t < self.t[-1]:
                    t_idx = np.where(self.t >= t)[0][0]
                else:
                    t_idx = int(-1)
                exec(f"{pname} = self.parameters['{pname}'][t_idx]")
            else:
                exec(f"{pname} = self.parameters['{pname}']")
        exec(f'{self.variables[0]}, {self.variables[1]} = u')
        out[0] = eval(self.model[0]) + self.input_func(t)[0]
        out[1] = eval(self.model[1]) + self.input_func(t)[0]
        return out

    def solve(self, x0, t=None, step_size=0.1, **kwargs):
        if t is None:
            t = self.t
        if self.solver == 'odeint':
            self.sol = odeint(self.system, x0, t, **kwargs)
            self.time_vals = t

    def plot_solution(self, save_fig=False, fig_fname='test.png', title=None, x_compare=None, compare_idx=0):

        if type(self.time_vals) != np.ndarray:
            raise AttributeError("Solution hasn't been evaluated yet!")

        if x_compare is None:
            for i in range(self.sol.shape[1]):
                plt.plot(self.time_vals, self.sol[:, i], label=self.variables[i])
        else:
            if type(compare_idx) is not list:
                compare_idx = [compare_idx]
            for _, idx in enumerate(compare_idx):
                ref_plot, = plt.plot(self.time_vals, x_compare[:, idx], label=self.variables[idx] + '_reference')
                color = ref_plot.get_color()
                plt.plot(self.time_vals, self.sol[:, idx], label=self.variables[idx], c=color, linestyle='--')

        plt.legend(loc='best')
        if title is None:
            plt.title(self.model_name)
        else:
            plt.title(title)
        plt.xlabel('t')
        if save_fig:
            plt.savefig(fig_fname)
            print(f'Saved figure to {fig_fname}')
            plt.close()
        else:
            plt.show()

    def plot_phase(self, t, x_lim=[-5, 5], y_lim=[-5, 5], x_density=10, y_density=10, quiver_scale=2,
                   x_label=None, y_label=None, save_fig=False, fig_fname='2D_system_phase_space.png',
                   plot_nullclines=True, get_equilibiria=True):

        # TODO: extend to 1D system making a slope field plot of solutions

        if x_label == None:
            x_label = self.variables[0]
        if y_label == None:
            y_label = self.variables[1]

        x = np.linspace(x_lim[0], x_lim[1], x_density)
        y = np.linspace(y_lim[0], y_lim[1], y_density)
        vect0 = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

        fig = plt.figure()

        #######################################
        # generate the phase space streamplot #
        #######################################

        for i in range(vect0.shape[0]):
            x0 = [vect0[i][0], vect0[i][1]]
            self.solve(x0=x0, t=t)
            sol = self.sol
            plt.quiver(sol[:-1, 0], sol[:-1, 1], sol[1:, 0] - sol[:-1, 0], sol[1:, 1] - sol[:-1, 1], scale_units='xy',
                       angles='xy', scale=quiver_scale, color='k', alpha=0.5)
            plt.plot(sol[:, 0], sol[:, 1], color='k', alpha=0.5)

        ############################################
        # optionally add nullclines and equilibira #
        ############################################
        if not self.dynamic_parameters:
            if plot_nullclines:
                if len(self.nullclines) == 0:

                    self.get_nullclines_and_jacobian()
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

            if get_equilibiria:
                if not plot_nullclines:
                    self.get_nullclines_and_jacobian()
                    print(f'found nullclines: {self.nullclines}')
                    x_length = x_lim[1] - x_lim[0]
                    xs = np.arange(x_lim[0], x_lim[1], x_length / 1000)
                    for _, pname in enumerate(self.parameters.keys()):
                        exec(f"{pname} = self.parameters['{pname}']")
                    nullcline_dict = self.parameters.copy()
                    nullcline_dict[f'{self.variables[0]}'] = xs
                    x_nullcline_values = eval(f'{self.nullclines[0]}', nullcline_dict)
                    y_nullcline_values = eval(f'{self.nullclines[1]}', nullcline_dict)

                # find crossings of nullclines
                sign_array = np.sign(x_nullcline_values - y_nullcline_values)
                if not len(sign_array.shape) == 0:
                    idx = np.argwhere(np.diff(sign_array)).flatten()
                    intersections = np.array([xs[idx], x_nullcline_values[idx]])


                    # stability analysis and equilibrium classification
                    n_intersect = intersections.shape[1]
                    intersect_stability = []


                    for _, pname in enumerate(self.parameters.keys()):
                        exec(f"{pname} = sy.symbols('{pname}')")

                    print('equilibria:')
                    for i in range(n_intersect):

                        # TODO: caution here is a numpy version problem, if a float x is returned as np.float(x) it will
                        #  cause trouble in sympy
                        local_jacobi = np.array(self.Jacobian.subs([(self.sympy_var_1,  intersections[0, i]),
                                                                    (self.sympy_var_2, intersections[1, i])]), dtype=float)
                        eigvals, eigvec = np.linalg.eig(local_jacobi)
                        stability = get_stability_2D(eigvals)
                        intersect_stability.append(stability)
                        print(f'{self.variables[0]} = {intersections[0, i]:.3f},  {self.variables[1]} ='
                              f' {intersections[1, i]:.3f}, {stability}')
                        if 'unstable' in stability:
                            plt.scatter(intersections[0, i], intersections[1, i], s=50, facecolors='none', edgecolors='k',
                                        zorder=10)
                        elif 'stable' in stability:
                            plt.scatter(intersections[0, i], intersections[1, i], s=50, c='k', zorder=10)
                        elif 'unknown' in stability:
                            plt.scatter(intersections[0, i], intersections[1, i], s=50, c='k', marker='star', zorder=10)
                        else:
                            plt.scatter(intersections[0, i], intersections[1, i], s=50, c='r', marker='square', zorder=10)
                else:
                    print('no equilibria found')

        else:
            print('Dynamic parameters found, "plot_equilibria" and "plot_nullcline" are ignored!')


        plt.xlim(0.95*x_lim[0], 0.95*x_lim[1])
        plt.ylim(0.95*y_lim[0], 0.95*y_lim[1])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(self.model_name + ' phase_space')

        # either save or show
        if save_fig:
            plt.savefig(fig_fname)
            print(f'Saved to {fig_fname}')
            plt.close()
        else:
            plt.show()

    def get_nullclines_and_jacobian(self):

        self.sympy_var_1, self.sympy_var_2 = sy.symbols(f'{self.variables[0]}, {self.variables[1]}')
        sympy_model = [self.model[k].replace('np.', '') for k in [0, 1]]

        x_expr = sy.parsing.sympy_parser.parse_expr(sympy_model[0], evaluate=False)
        x_nullcline = sy.solve(x_expr, self.sympy_var_2)
        if x_nullcline == []:
            x_nullcline = sy.solve(x_expr, self.sympy_var_1)
        if len(x_nullcline) > 1:
            warnings.warn(f'x_nullcline has multiple solutions for y: {x_nullcline}')
        y_expr = sy.parsing.sympy_parser.parse_expr(sympy_model[1], evaluate=False)
        y_nullcline = sy.solve(y_expr, self.sympy_var_2)

        self.nullclines = [str(x_nullcline[0]), str(y_nullcline[0])]

        # calculate Jacobian
        for _, pname in enumerate(self.parameters.keys()):
            exec(f"{pname} = sy.symbols('{pname}')")

        system_matrix = sy.Matrix([x_expr.subs(self.parameters), y_expr.subs(self.parameters)])
        self.Jacobian = system_matrix.jacobian([self.sympy_var_1, self.sympy_var_2])

    def parameter_fit(self, target_series, x0=[0, 0], t=None, t_end=None, t_start=None, parameter='a',
                      variables=['x'], parameter_bounds=[0, 1], method='hierarchical_zoom', eps=0.1,
                      max_iter=100, verbose=False):

        # TODO: implement  least squares from scipy with the 3 methods it has for goal function nrmse(target_series, x)

        # set up input
        steps = target_series.shape[0]
        if t is None:
            if t_end is not None and t_start is not None:
                t = np.linspace(t_start, t_end, steps)
            else:
                raise NotImplementedError('Please give information about time, either through passing t or t_start and'
                                          ' t_end in arguments!')
        if type(variables) is not list:
            variables = [variables]

        if len(variables) == 1:
            target_series = target_series[:, np.newaxis]

        sol_idxs = [self.variables.index(k) for k in variables]

        if method == 'hierarchical_zoom':
            lower_bound = parameter_bounds[0]
            upper_bound = parameter_bounds[1]

            i = 0
            min_error = 1.
            while i+1 < max_iter:
                n_grid = 20
                param_values = np.random.uniform(lower_bound, upper_bound, n_grid)
                error = np.zeros_like(param_values)
                for j in range(n_grid):
                    self.parameters[parameter] = param_values[j]
                    self.solve(x0=x0, t=t)
                    x = self.sol[:, sol_idxs]
                    error[j] = nrmse(target_series, x)
                min_error = np.min(error)
                min_error_idx = np.argmin(error)
                if verbose==2:
                    print(f'parameter guess for {parameter} = {param_values[min_error_idx]:.5f}')
                    print(f'nrmse = {min_error:.5f}')
                if min_error < eps:
                    self.parameters[parameter] = param_values[min_error_idx]
                    self.fit_error = min_error
                    if verbose > 0:
                        print(f'converged with nrmse = {min_error:.5f} and parameter {parameter} ='
                              f' {param_values[min_error_idx]:.5f}')
                    break
                # get new bounds for next iteration
                p_new = param_values[min_error_idx]
                delta = upper_bound - lower_bound
                lower_bound = max(lower_bound, p_new - 0.5 * delta)
                upper_bound = min(upper_bound, p_new + 0.5 * delta)
                i += 1

            if i+1 == max_iter:
                print(f'maxmimum iterations ({max_iter}) reached')
                self.fit_error = min_error

    def parameter_ramp(self, return_to_zero=True, t_start=None, t_end=None, t_turn=5, p_start=0, p_end=1.0,
                       parameter=None):

        if self.t is None:
            self.t = np.linspace(t_start, t_end, 1000)

        if t_start != None:
            t_start_idx = np.where(self.t > t_start)[0][0]
        else:
            t_start_idx = 0

        if t_end != None:
            if t_end >= self.t[-1]:
                t_end_idx = self.t.shape[0] - 1
            else:
                t_end_idx = np.where(self.t > t_end)[0][0]
        else:
            t_end_idx = self.t.shape[0] - 1

        if not return_to_zero:
            n_time_steps = t_end_idx - t_start_idx
            p_vals = np.zeros_like(self.t)
            p_vals[:t_start_idx] = p_start
            p_vals[t_start_idx:t_end_idx] = np.linspace(p_start, p_end, n_time_steps)
            p_vals[t_end_idx:] = p_end
        else:
            t_turn_idx = np.where(self.t > t_turn)[0][0]
            n_time_steps_1 = t_turn_idx - t_start_idx
            n_time_steps_2 = t_end_idx - t_turn_idx
            p_vals = np.zeros_like(self.t)
            p_vals[:t_start_idx] = p_start
            p_vals[t_start_idx:t_turn_idx] = np.linspace(p_start, p_end, n_time_steps_1)
            p_vals[t_turn_idx:t_end_idx] = np.linspace(p_end, p_start, n_time_steps_2)
            p_vals[t_end_idx:] = p_start

        self.parameters[parameter] = p_vals

class General1DSystem(General2DSystem):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.model_name == None:
            self.model_name = '1D_system'
        if self.model == None:
            self.model = 'x'


    def system(self, u, t):
        for _, pname in enumerate(self.parameters.keys()):
            exec(f"{pname} = self.parameters['{pname}']")
        exec(f'{self.variables[0]} = u')
        out = eval(self.model) + self.input_func(t)
        return out

    def solve(self, x0, t, step_size=0.1, **kwargs):
        if self.solver == 'odeint':
            self.sol = odeint(self.system, x0, t, **kwargs)
            self.time_vals = t


def phase_portrait(v1, v2, diffeq,
                   *params,
                   color='black',
                   ax=None):
    """Plots phase portrait given a dynamical system

    Given a dynamical system of the form dXdt=f(X,t,...), plot the phase portrait of the system.

    Parameters
    ----------

    v1 : array
        The range of the first variable

    v2 : array
        The range of the second variable

    diffeq : function
        The function dXdt = f(X,t,...)

    params:
        System parameters to be passed to diffeq

    ax : pyplot plotting axes
        Optional existing axis to pass to function

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """
    if (ax is None):
        fig, ax = plt.subplots(figsize=(12, 8))  # Troy edit to make plot bigger

    V1, V2 = np.meshgrid(v1, v2)  # create rectangular grid with points
    t = 0  # start time
    u, w = np.zeros(V1.shape), np.zeros(V2.shape)  # initiate values for quiver arrows
    NI, NJ = V1.shape

    for i in range(NI):
        for j in range(NJ):
            xcoord = V1[i, j]
            ycoord = V2[i, j]
            vprime = diffeq([xcoord, ycoord], t, *params)
            u[i, j] = vprime[0]
            w[i, j] = vprime[1]

    r = np.sqrt(u**2 + w**2)
    r = np.where(r == 0, 1, r)

    Q = ax.quiver(V1, V2, u / r, w / r,  # TROY EDIT using ax instead of plt to fix last code cell output
                  color=color)

    return ax


def slope_field(t, x, diffeq, units='xy', angles='xy', scale=None, color='black', ax=None, **args):
    """Plots slope field given an ode

    Given an ode of the form: dx/dt = f(t, x), plot a slope field (aka direction field) for given t and x arrays.
    Extra arguments are passed to matplotlib.pyplot.quiver

    Parameters
    ----------

    t : array
        The independent variable range

    x : array
        The dependent variable range

    diffeq : function
        The function f(t,x) = dx/dt

    args:
        Additional arguments are aesthetic choices passed to pyplot.quiver function

    ax : pyplot plotting axes
        Optional existing axis to pass to function

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """
    if (ax is None):
        fig, ax = plt.subplots()
    if scale is not None:
        scale = 1 / scale
    T, X = np.meshgrid(t, x)  # create rectangular grid with points
    slopes = diffeq(T, X)
    dt = np.ones(slopes.shape)  # dt = an array of 1's with same dimension as diffeq
    dxu = slopes / np.sqrt(dt ** 2 + slopes ** 2)  # normalize dx
    dtu = dt / np.sqrt(dt ** 2 + slopes ** 2)  # normalize dt
    ax.quiver(T, X, dtu, dxu,  # Plot a 2D field of arrows
              units=units,
              angles=angles,  # each arrow has direction from (t,x) to (t+dt, x+dx)
              scale_units='x',
              scale=scale,  # sets the length of each arrow from user inputs
              color=color,
              **args)  # sets the color of each arrow from user inputs

    return ax