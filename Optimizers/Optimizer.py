import warnings
import numpy as np
import sympy as sy
from scipy.integrate import odeint
from Utils import get_stability_2D, nrmse
import matplotlib.pyplot as plt
import time
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
    # mutation_single, crossover, mutation

    # evaluation is not from GA toolbox

    def crossover(self, X, n):
        """
        Perform n crossover operations on a population X.

        Parameters:
        X (ndarray): Population matrix of shape (population_size, parameter_size)
        n (int): Number of crossover pairs

        Returns:
        ndarray: E New population after crossover (2*n individuals)
        """
        x, y = X.shape  # population size and parameter size
        E = np.zeros((2 * n, y))

        for i in range(n):
            # Select two distinct chromosomes
            r = np.random.choice(x, size=2, replace=False)
            A = X[r[0]].copy()
            B = X[r[1]].copy()

            # Select cut point
            c = np.random.randint(1, y)

            # Perform crossover, putting back part from A to B and vise versa
            A_back_part = A[c:].copy()
            A[c:] = B[c:]
            B[c:] = A_back_part

            # Store new chromosomes
            E[2 * i] = A
            E[2 * i + 1] = B

        return E

    def mutationV(self, P, lowchance, highchance, LR, UR):
        """
        Applies mutation to a population based on mutation probability.

        Parameters:
        - P: 2D array of shape (n_pop, n_param), ranked population
        - lowchance: mutation chance for best individual
        - highchance: mutation chance for worst individual
        - LR: lower bound (scalar or array)
        - UR: upper bound (scalar or array)

        Returns:
        - mutateP: mutated population
        """
        mutateP = P.copy()
        n_pop, n_param = mutateP.shape
        mutateChance = np.linspace(lowchance, highchance, n_pop)

        # Create mutation mask
        mask = np.random.rand(n_pop, n_param) < mutateChance[:, np.newaxis]

        # Generate new random values for mutation
        mutation = self.population(n_pop, n_param, LR, UR)

        # Apply mutation
        mutateP[mask] = mutation[mask]

        return mutateP

    def population(self, N, nParams, LR, UR):
        P = np.zeros((N, nParams))
        for i in range(nParams):
            P[:, i] = (UR[i] - LR[i]) * np.random.rand(N) + LR[i]
        return P

    def fitness_function(self, y, h_out):
        """
        Compute the fitness function between a goal function y and a model output h_out
        :param y: np.ndarray, goal function
        :param h_out: np.ndarray, model output
        :return: fit: np.ndarray, fitness value
        """
        difference = y-h_out
        fit = np.var(difference)/np.var(y)
        return fit

    def gradient_search(self, P, r, conf, stop_crit):
        """
        Perform a gradient search for a population P on a model conf
        :param P:
        :param r:
        :param conf:
        :param stop_crit:
        :return:
        """

    def gauss_newton_slow(self, op, Para_E_test, r_test, func, y_goal, reg0, reg1, steps, loop, tol, LR, UR, fit_crit):
        """
        Performs iterative Gauss-Newton optimization with Levenberg regularization.

        Parameters:
        - op: optimization direction (positive or negative)
        - Para_E_test: initial parameter vector (1D array)
        - r_test: initial residual
        - func: cost function
        - y_goal: target output
        - reg0, reg1: regularization parameters
        - steps: number of candidate steps
        - loop: max number of iterations
        - tol: tolerance for stopping
        - LR, UR: lower and upper bounds for gradient repair
        - fit_crit: fitness criterion for stopping

        Returns:
        - fit_after_g: final fitness value
        - Para_E_after_g: optimized parameter vector
        - error_after_g: final residual
        """
        j = 1
        fit_ = []
        Para_E_ = []
        error_ = []

        while j <= loop:
            print(f"[{j}/{loop}] ", end="")
            J = self.NMM_diff_A_lfm(Para_E_test, r_test, func, y_goal)
            Para_E_new_group = self.multi_lavenberg_regulization(steps, reg0, reg1, Para_E_test, J, r_test, LR, UR)

            fit_grp, error_grp = self.evaluation(Para_E_new_group, func, y_goal)
            Para_E_new, fit_new, error_new = self.selection_best(Para_E_new_group, fit_grp, error_grp, 1, op)

            r_test = error_new
            Para_E_test = Para_E_new.flatten()

            print(fit_new)
            fit_.append(fit_new)
            Para_E_.append(Para_E_test.copy())
            error_.append(r_test.copy())
            j += 1

            if len(fit_) > 1 and op * (fit_[-1] - fit_[-2]) < tol:
                print(f"Quit: improvement < tol({tol})")
                break

        if j == loop:
            Para_E_after_g, fit_after_g, error_after_g = self.selection_best(np.array(Para_E_), np.array(fit_),
                                                                        np.array(error_), 1, op)
            Para_E_after_g = Para_E_after_g.flatten()
        else:
            Para_E_after_g = Para_E_test
            fit_after_g = fit_new
            error_after_g = r_test

        return fit_after_g, Para_E_after_g, error_after_g

    def NMM_diff_A_lfm(self, parameter, h_output, myfunc, y_goal):
        """
        Computes the Jacobian matrix using finite differences.

        Parameters:
        - para: current parameter vector (1D array)
        - houtput: current output from myfunc(para, y_goal)
        - myfunc: function to evaluate
        - y_goal: target output for myfunc

        Returns:
        - j: Jacobian matrix (samples x parameters)
        """
        h = 1e-6
        parameter_pert = parameter + h
        j = np.zeros((len(h_output), len(parameter)))

        for i in range(len(parameter)):
            parameter_update = parameter.copy()
            parameter_update[i] = parameter_pert[i]
            h_output_new = myfunc(parameter_update, y_goal)
            j[:, i] = (h_output_new - h_output) / h

        j[np.isnan(j)] = 0
        j[np.isinf(j)] = 0

        return j

    def evaluation(self, X, func, y):
        """
        GA toolbox evaluation function that helps evaluating a simulation function 'func' over an array 'X' of values
        :param X: parameter values
        :param func: simulation function
        :param y: reference
        :return: fits: fit values (errors), h_outs: output values of evaluated functions
        """
        x_shape = X.shape[0]
        fits = np.zeros(x_shape)
        h_outs = np.zeros_like(fits)

        for j in range(x_shape):
            P = X[j]
            start_time = time.time()
            h_out = func(P)
            fit = nrmse(h_out, y)
            end_time = time.time()
            print(f' simulation time: {end_time:.5f} --> {j+1}/{x_shape}')
            fits[j] = fit
            h_outs[j] = h_out

        return fits, h_outs

    def multi_lavenberg_regulization(self, n, reg0, reg1, Para_E, J, h_output, LR, UR):
        """
        Generates n updated parameter sets using Levenberg regularization.

        Parameters:
        - n: number of populations to generate
        - reg0, reg1: min and max regularization exponents
        - Para_E: current parameter vector (1D array)
        - J: Jacobian matrix
        - h_output: residual vector
        - LR, UR: lower and upper bounds for gradient repair

        Returns:
        - Y: 2D array of shape (n, len(Para_E)) with updated parameter sets
        """
        Y = np.zeros((n, len(Para_E)))
        reg = 10 ** np.linspace(reg0, reg1, n)

        for i in range(len(reg)):
            try:
                D = np.linalg.pinv(J.T @ J + reg[i] * np.eye(len(Para_E)))
            except:
                # TODO: try if this error is error important and find out what exception is necessary
                return Y  # return current Y if inversion fails

            d = -D @ J.T @ h_output
            if np.isnan(d).any():
                continue  # skip this iteration if invalid update

            Para_E_new = Para_E + d
            Y[i, :] = self.gradient_repair(Para_E_new, LR, UR)

        return Y

    def gradient_repair(self, Para_E, LR, UR):
        """
        Function that 'repairs' a parameter selection given upper and lower bounds
        Upper and lower bounds are replaced by new values sampled in bounds or by boundary values for inhomogeneous
        :param Para_E:
        :param LR:
        :param UR:
        :return:
        """
        if len(UR) == 1:
            upper_cut = Para_E > UR
            under_cut = Para_E < LR
            Para_E[upper_cut] = self.population(1, Para_E[upper_cut].shape[0], LR, UR)
            Para_E[under_cut] = self.population(1, Para_E[under_cut].shape[0], LR, UR)
        else:  # individual constraints
            for i in range(Para_E.shape[0]):
                if Para_E[i] > UR[i]:
                    Para_E[i] = UR[i]
                elif Para_E[i] < LR[i]:
                    Para_E[i] = LR[i]
        return Para_E