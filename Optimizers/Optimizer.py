import warnings
import numpy as np
import sympy as sy
from scipy.integrate import odeint
from Utils import get_stability_2D, nrmse
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')

class Optimizer():
    def __init__(self, parameters):
        self.opt_parameters = np.zeros((2))
        # self.simulate = lambda self.opt_paramters[0]: x

class Hierarchical_Random(Optimizer):
    def __init__(self, parameters):
        super().__init__(parameters=parameters)

        self.t = np.zeros(100)
        self.max_iter = 10
        self.noise_term = 0.005
        self.eps = 0.01
        self.min_errors = []
        self.optimum = None

        self.opt_idxs = []
        self.n_grid = 20
        self.x_out = None
        self.simulate = None
        self.y = None
        self.model_parameters = None
        self.simulation_class = None
        self.bounds = None

        self.__dict__.update(parameters)
        self.parameters = parameters

        assert type(self.y) == np.ndarray, 'please provide a validation function!'
        assert self.simulate != None, 'please provide a model'
        assert self.model_parameters != None, 'please provide a parameter names'
        assert self.bounds != None, 'please provide a parameter ranges'

        self.n_param = len(self.model_parameters)
        self.lower_bound = np.zeros(self.n_param)
        self.upper_bound = np.zeros(self.n_param)
        for i in range(self.n_param):
            self.lower_bound[i] = self.bounds[i][0]
            self.upper_bound[i] = self.bounds[i][1]

        self.error = np.ones((self.max_iter, self.n_grid))
        self.min_error_idxs = np.zeros(self.max_iter)
        t_shape = self.simulation_class.t.shape[0]
        self.x_vals = np.zeros((self.max_iter, self.n_grid, t_shape))
        self.opt_parameters = np.zeros((self.max_iter, self.n_grid, self.lower_bound.shape[0]))

    def run(self):
        previous_min_error = 1

        for i in tqdm(range(self.max_iter)):
            param_values = np.zeros((self.n_param, self.n_grid))
            keywords = self.parameters
            for j in range(self.n_param):
                param_values[j] = np.random.uniform(self.lower_bound[j], self.upper_bound[j], self.n_grid)

            for k in range(self.n_grid):
                self.opt_parameters[i, k] = param_values[:, k]

                for l in range(self.n_param):
                    keywords[self.model_parameters[l]] = param_values[l, k]
                keywords['y'] = self.y
                keywords['idx'] = f'{i}_{k}'
                # keywords.update(...)
                if self.simulation_class == None:
                    x = self.simulate(**keywords)
                elif self.x_out == None:
                    self.simulation_class.__init__(parameters=keywords)
                    self.simulate()
                else:
                    self.simulation_class.__init__(parameters=keywords)
                    self.simulate()
                    # exec(f'x = self.simulation_class.{self.x_out}')  # idk this doesn't work atm
                    x = self.simulation_class.mass_model_v_out
                self.x_vals[i, k] = x
                # TODO: make this work with independent functions eventually
                # self.error[i, k] = nrmse(self.y, x)
                self.error[i, k] = self.simulation_class.error

            min_error = np.nanmin(self.error[i])

            min_error_idx = (i, np.nanargmin(self.error[i]))
            self.min_error_idxs[i] = min_error_idx[1]
            self.min_errors.append(min_error)
            self.opt_idxs.append(min_error_idx)
            print('#########################################################################')
            print(f'error: {min_error:.5f}, at index {min_error_idx}')
            print(f'{param_values[:, min_error_idx[1]]}')
            print('#########################################################################')

            if min_error < self.eps:
                print(f'error: {min_error:.4f}')
                self.optimum = param_values
                break
            if i > 0:
                previous_min_error = np.min(np.array(self.min_errors[:i]))
            if min_error < previous_min_error - self.noise_term:
                # contract region in parameter space if error was smaller
                print(f'new min error, updating parameters')

                # get new bounds for next iteration
                p_new = param_values[:, min_error_idx[1]]
                delta = self.upper_bound - self.lower_bound
                for j in range(self.n_param):
                    self.lower_bound[j] = max(self.lower_bound[j], p_new[j] - 0.5 * delta[j])
                    self.upper_bound[j] = min(self.upper_bound[j], p_new[j] + 0.5 * delta[j])
            else:
                print(f'error not smaller than {previous_min_error:.4f}-{self.noise_term}')
        if self.optimum == None:
            self.optimum = param_values