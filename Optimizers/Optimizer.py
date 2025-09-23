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
                keywords['nykamp_parameters']['connectivity_matrix'] = np.array([[param_values[-1, k]]])  # hotfix!
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
            print('\n#########################################################################')
            print(f'error: {min_error:.5f}, at index {min_error_idx}')
            print(f'{param_values[:, min_error_idx[1]]}')
            print('#########################################################################')

            if min_error < self.eps:
                print(f'error: {min_error:.4f}')
                self.optimum = param_values[:, min_error_idx[1]]
                print(f'optimal values: {self.optimum}')
                break
            if i > 0:
                previous_min_error = np.min(np.array(self.min_errors[:i]))
            if min_error < previous_min_error - self.noise_term:
                # contract region in parameter space if error was smaller
                print(f'new min error, updating parameters')

                # get new bounds for next iteration
                p_new = param_values[:, min_error_idx[1]]
                self.optimum = p_new
                print(f'optimal values: {self.optimum}')
                delta = self.upper_bound - self.lower_bound
                for j in range(self.n_param):
                    self.lower_bound[j] = max(self.lower_bound[j], p_new[j] - 0.5 * delta[j])
                    self.upper_bound[j] = min(self.upper_bound[j], p_new[j] + 0.5 * delta[j])
            else:
                print(f'error not smaller than {previous_min_error:.4f}-{self.noise_term}')

class GA(Optimizer):
    def __init__(self, parameters):
        super.__init__(parameters=parameters)

        self.model_parameters = None
        self.simulation_class = None
        self.goal_func = None
        self.op = 1
        self.n_iter = 50
        self.reference = None
        self.bounds = None

        LR = self.bounds[:, 0]
        UR = self.bounds[:, 1]
        nParams = len(LR)
        GA_counter = []
        
        conf = {'UR': UR, 'LR': LR, 'op': self.op, 'func':self.simulation_class, 'goal_func':self.goal_func,
                'gLoop':10, 'gL': -12, 'gU': 12, 'gTol': 0.01}
        conf['gT'] = abs(conf['gU'] - conf['gL']) + 1

        ################################################################
        N1 = 60  # population size
        N2 = 100  # crossover, number of pair to crossover
        N3 = 100  #  mutation, number of pairs to mutate
        tg = 1 # total generations

        K = 0  # history of[average cost, best cost]
        KP = 0 # history of[best solution]
        KS = 0 # history of[best cost]
        w = 1  # counter
        j = 1  # counter

        ######## initialization #########
        print('======== Initialization ========')
        P = self.population(N1, nParams, LR, UR)  # generate[60 x nParams] random solutions
        # P = [P, solution_ini]  # add pre - selected solutions
        # TODO: check if this is necessary later
        E, R = self.evaluation(P, self.simulation_class) # E: evaluation fitness, R: residual, error
        P, E, R = self.selection_best(P, E, R, N1, self.op)
        R1 = R[:, 0]
        print('done')
        print([f'Minimum cost: ', {E(1)}])
        print('================================')
        E_crit = E[0]

        ################## loop ######################
        for j in range(self.n_iter):
            print('======= Gradient search ========')
            Para_E_grd, E_grd, R_grd = self.gradient_search[P[0], R1, conf, E_crit]
            # replace
            if self.op * E_grd > self.op * E(1):
                P[0] = Para_E_grd
                E[0] = E_grd
                R[0] = R_grd
            print('done')

            print('======= single-parameter mutation ========')
            P_ = self.mutation_single(P[1,:], LR, UR)

            [E_, R_] = self.evaluation(P_, conf['func'], self.ref)
            print('done')

            print('======= Gradient search ========')
            for i in range(E_.shape[0]):
                print('[#d/#d] cost: #f\n', i, len(E_), E_(i))
                Para_E_grd[i,:], E_grd[i], R_grd[:, i] = self.gradient_search(P_[i,:], R_[:, i], conf, E_crit)

            # replace
            index = self.op * E_grd > self.op * E_
            P_[index,:] = Para_E_grd[index,:] #  update gradient mutation
            E_[index] = E_grd(index)
            R_[:, index] = R_grd[:, index]
            P = [P, P_]  # [(60 + nParams) x nParams] solutions
            E = [E, E_]  # [1 x(60 + nParams)] cost
            R = [R, R_]  # [timepoints x(60 + nParams)] residual
            print('done')

            # # # # # # # # # #
            # add to show, delete later
            _, E_show, _ = self.selection_best(P, E, R, 1, self.op)
            print([f'best after gradient: ', {E_show}])
            # # # # # # # # # #

            # GA
            print('GA search...')
            # matlab specific
            # TODO: solve with vstack or concatenate
            P[-1 + 1:-1 + N1,:] = self.mutationV(P[:N1,:],0.1, 0.9, LR, UR) # + N1 solutions
            P[-1 + 1:-1 + 2 * N2,:] = self.crossover(P, N2) # + N2 * 2 solutions
            P[-1 + 1:-1 + 2 * N3,:] = self.mutation(P, N3) # + N3 * 2 solutions

            E_, _, _ = self.evaluation(P[N1 + nParams + 1:-1,:], conf['func'], self.ref)
            E = [E, E_] # cost[1 x (N1 + N2 + N3) * 2 + nParams]

            # selection
            P, E = self.selection_uniq(P, E, N1, N1, self.op, LR, UR) # select N1 solutions
            _, R1, _ = self.evaluation(P[1,:], conf['func'], self.ref) # R1: residual of best solution
            print('done')

            K[w, 1] = sum(E) / N1  # average  cost(for plot)
            K[w, 2] = E(1)  # best cost(for plot)
            KP[w, 1: nParams] = P[1, 1: nParams]  # save best
            KS[w] = E(1) # save best
            E_crit = E(1)
            print('========')
            print([f'current best Loss: {KS[w]}'])
            print('========')
            gof = self.fitness_function(self.ref.y0, R1)

            print('========')
            print([f'current best R2: {gof}'])
            print('========')
            # # # # # # #
            # add to show, delete later
            if E_show > E[0]:
                print('GA works')
                GA_counter[w] = 1
            else:
                print("GA doesn't work")
                GA_counter[w] = 0
            # # # # # # # # # #

            # stop: good fit
            if KS[-1] < 0.01:
                break
    # self.population, fitness_function, evaluation, selection_uniq, mutationV, selection_best, gradient_search,
    # mutation_single, GA_counter, crossover, mutation