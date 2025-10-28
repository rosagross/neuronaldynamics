import warnings
import numpy as np
import sympy as sy
from scipy.integrate import odeint
from Utils import get_stability_2D, nrmse, t_format
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import time
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')

class Optimizer():
    def __init__(self, parameters):
        self.opt_parameters = np.zeros((2))
        self.optimum = None
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
        assert self.simulate != None, 'please provide a model!'
        assert self.model_parameters != None, 'please provide parameter names!'
        assert self.bounds != None, 'please provide parameter ranges!'

        self.n_param = len(self.model_parameters)
        self.lower_bound = np.zeros(self.n_param)
        self.upper_bound = np.zeros(self.n_param)
        for i in range(self.n_param):
            self.lower_bound[i] = self.bounds[i][0]
            self.upper_bound[i] = self.bounds[i][1]

        self.error = np.ones((self.max_iter, self.n_grid))
        self.min_error_idxs = np.zeros(self.max_iter)
        if hasattr(self.simulation_class, 't'):
            t_shape = self.simulation_class.t.shape[0]
            self.x_vals = np.zeros((self.max_iter, self.n_grid, t_shape))
        else:
            self.x_vals = np.zeros((self.max_iter, self.n_grid))
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
                # keywords['nykamp_parameters']['connectivity_matrix'] = np.array([[param_values[-1, k]]])  # hotfix!
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
                    x = eval(f'self.simulation_class.{self.x_out}')
                self.x_vals[i, k] = x
                # self.error[i, k] = nrmse(self.y, x)
                if hasattr(self.simulation_class, 'error'):
                    self.error[i, k] = self.simulation_class.error
                else:
                    #TODO: extend to other fit functions
                    self.error[i, k] = nrmse(self.y, x)
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
        super().__init__(parameters=parameters)
        self.parameters = parameters
        self.model_parameters = None
        self.simulation_class = None
        self.simulation = None
        self.op = 1
        self.n_iter = 50
        self.x_out = 'y'
        self.reference = None
        self.tolerance = 0.05
        self.verbose = 0

        self.bounds = None
        self.N1 = 60  # population size
        self.N2 = 100  # crossover, number of pair to crossover
        self.N3 = 100  # mutation, number of pairs to mutate
        self.__dict__.update(parameters)
        self.errors = None
        self.parameter_evolution = None

        assert isinstance(self.reference, (np.ndarray, int, float)), 'please provide a reference value!'
        assert self.model_parameters != None, 'please provide parameter names!'
        assert self.bounds != None, 'please provide parameter ranges!'
        assert self.simulation != None or self.simulation_class != None, ('please provide a simulation function or a'
                                                                          ' class with a simulation method!')
        assert len(self.model_parameters) == len(self.bounds), ("The number of parameters must match the number of"
                                                              " specified bounds!")
        self.n_param = len(self.model_parameters)
        if type(self.bounds) == list:
            self.bounds = np.array(self.bounds)

        if hasattr(self.simulation_class, 't'):
            self.t_shape = self.simulation_class.t.shape[0]
        else:
            self.t_shape = 1

        if self.simulation_class !=None:
            self.simulation_function = self.simulation_class.simulate
        else:
            self.simulation_function = self.simulate

    def run(self):
        """
        Run the optimizer
        """
        LR = self.bounds[:, 0]
        UR = self.bounds[:, 1]
        nParams = len(LR)
        GA_counter = []
        
        conf = {'UR': UR, 'LR': LR, 'op': self.op, 'func': self.simulation_function,
                'gLoop': 10, 'gL': -12, 'gU': 12, 'gTol': self.tolerance}
        conf['gT'] = abs(conf['gU'] - conf['gL']) + 1

        ################################################################
        K = []  # history of[average cost, best cost]
        KP = [] # history of[best solution]
        KS = [] # history of[best cost]
        w = 0  # counter
        j = 0  # counter

        ######## initialization #########
        print('======== Initialization ========')
        P = self.population(self.N1, nParams, LR, UR)  # generate[60 x nParams] random solutions
        # P = [P, solution_ini]  # add pre - selected solutions
        E, _, R = self.evaluation(P, self.reference) # E: evaluation fitness, R: residual, error
        P, E, R = self.selection_best(P, E, R, self.N1, self.op)
        R1 = R[0]
        if isinstance(R1, (int, float)):
            R1 = np.array([[R1]])
        print('done')
        print(f'Minimum cost: {E[0]:.5}')
        print('================================')
        E_crit = E[0]

        ################## loop ######################
        for j in range(self.n_iter):
            print('======= Gradient search ========')
            Para_E_grd, E_grd, R_grd = self.gradient_search(P[0], R1, conf, E_crit)
            # replace
            if self.op * E_grd > self.op * E[0]:
                P[0] = Para_E_grd
                E[0] = E_grd
                R[0] = R_grd
            print('done')

            print('======= single-parameter mutation ========')
            P_ = self.mutation_single(P[0, :], LR, UR)

            E_, _, R_ = self.evaluation(P_, self.reference)
            print('done')

            print('======= Gradient search ========')
            n_search = E_.shape[0]
            E_grd_new = np.zeros(n_search)
            R_grd_new = np.zeros((n_search, R_.shape[-1]))
            E_grd_new[:E_grd.shape[0]] = E_grd
            R_grd_new[0, :R_grd.shape[-1]] = R_grd
            E_grd = E_grd_new
            R_grd = R_grd_new
            P_shape = P_.shape
            Para_E_grd_new = np.zeros((P_shape))
            Para_E_grd_new[:P_.shape[0]] = P_
            Para_E_grd = Para_E_grd_new
            for i in range(E_.shape[0]):
                if self.verbose > 0: print(f'[{i+1}/{len(E_)}] cost: {E_[i]:.5f}\n')
                P_i, E_i, R_i = self.gradient_search(P_[i, :], R_[i], conf, E_crit)
                if isinstance(R_i, np.ndarray) and len(R_i.shape) > 1:
                    R_i = R_i.flatten()
                Para_E_grd[i, :], E_grd[i], R_grd[i] = P_i, E_i, R_i

            # replace
            index = self.op * E_grd > self.op * E_
            P_[index, :] = Para_E_grd[index, :] #  update gradient mutation
            E_[index] = E_grd[index]
            R_[index] = R_grd[index]
            P = np.vstack((P, P_))  # [(60 + nParams) x nParams] solutions
            E = np.hstack((E, E_))
            if len(R.shape) > 1 : # [1 x(60 + nParams)] cost
                R = np.vstack((R, R_))  # [timepoints x(60 + nParams)] residual
            else:
                R = np.hstack((R, R_))
            print('done')

            # # # # # # # # # #
            # add to show, delete later
            P_show, E_show, _ = self.selection_best(P, E, R, 1, self.op)
            print([f'best after gradient: {E_show} at param set {P_show}'])
            # # # # # # # # # #

            # GA
            print('GA search...')
            # matlab specific
            mutV_marker = self.N1
            crossover_marker = self.N1+2*self.N2
            mut_marker = self.N1+2*self.N2+2*self.N3
            P_add = np.zeros((self.N1+2*self.N2+2*self.N3, P.shape[1]))
            P_add[:mutV_marker, :] = self.mutationV(P[:self.N1, :], 0.1, 0.9, LR, UR)  # + N1 solutions
            P_add[mutV_marker:crossover_marker, :] = self.crossover(P, self.N2)  # + N2 * 2 solutions
            P_add[crossover_marker:mut_marker, :] = self.mutation(P, self.N3)  # + N3 * 2 solutions
            P = np.vstack([P, P_add])

            E, _, _ = self.evaluation(P, self.reference) # was P[self.N1 + nParams + 1:, :] which is definetley wrong

            # selection
            P, E = self.selection_uniq(P, E, self.N1, self.N1, self.op, LR, UR) # select N1 solutions
            # _, R1, _ = self.evaluation(P[1,:], conf['func'], self.reference) # R1: residual of best solution
            print('done')

            K.append([sum(E) / self.N1, E[0]])  # average  cost(for plot) and  best cost(for plot)
            KP.append(P[0])  # save best
            KS.append(E[0]) # save best
            E_crit = E[0]
            print('========')
            print([f'current best Loss: {KS[w]}'])
            print('========')
            # gof = self.fitness_function(self.reference.y0, R1)
            #
            # print('========')
            # print([f'current best R2: {gof}'])
            # print('========')
            # # # # # # # #
            # add to show, delete later
            if E_show > E[0]:
                print('GA works')
                GA_counter.append(1)
            else:
                print("GA doesn't work")
                GA_counter.append(0)
            # # # # # # # # # #
            w+=1

            # stop: good fit
            if KS[-1] < 0.01:
                break
        print(f'best param set {KP[-1]} with error: {KS[-1]}')
        self.optimum = KP[-1]
        self.errors = KS
        self.parameter_evolution = np.array(KP)

<<<<<<< HEAD
    def anime_fit(self):
        assert self.errors != None, 'error'
=======
    def plot_fit(self):
        assert self.errors != None, "please run the optimizer first!"
>>>>>>> upstream/main
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.array(self.errors)*1e-22)
        ax.set_xlabel('# iteration')
        ax.set_ylabel('fit error')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(tck.MultipleLocator())
        plt.show()

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

    def mutation(self, X, n):
        """
            Perform mutation on a population of chromosomes.

            Parameters:
            X (ndarray): Population matrix of shape (x, y)
            n (int): Number of chromosome pairs to mutate

            Returns:
            ndarray: Mutated chromosomes, shape (2*n, y)
            """
        x, y = X.shape
        E = np.zeros((2 * n, y))

        for i in range(n):
            r = np.random.randint(0, x, size=2)
            while r[0] == r[1]:
                r = np.random.randint(0, x, size=2)

            A = X[r[0]].copy()
            B = X[r[1]].copy()
            c = np.random.randint(0, y)

            A[c], B[c] = B[c], A[c]

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
        """
        Function that creates population of random values for parameters within given upper and lower bounds
        :param N: Numbe of samples in population
        :param nParams: number of parameters (dimensionality)
        :param LR: lower boundary
        :param UR: upper boundary
        :return: P: population of random values
        """
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

    def selection_best(self, P, E, R, n_out, op=-1):
        """
        function that finds the best (fittest) parameter set from a population of parameter sets
        :param P: Input Population of parameter sets,(n_pop x n_parameter)
        :param E: Fitness values of parameter sets, (n_pop x n_parameter)
        :param R: residual, y-houtput  (n_data sample x n_pop)
        :param n_out: population size, the number of populations want to be returned
        :param op: default: -1, select minimum, 1 select maximum
        :return:  P_sorted : selected population,
                  E_sorted : the retrun of fitness of YY1
                  R_sorted : the return of residual of YY1
        """

        E = op*E
        # sort from high to low , get the best
        index = np.argsort(E)
        P_sorted = P[index]
        E_sorted = E[index]
        # R_sorted = R[:, index]
        #
        # return P_sorted[:n_out], E_sorted[:n_out], R_sorted[:, :n_out]
        R_sorted = R[index]
        return P_sorted[:n_out], E_sorted[:n_out], R_sorted[:n_out]

    def gradient_search(self, P, r, conf, stop_crit):
        """
        Perform a gradient search for a population P on a model conf
        run gradient search on the solutions
        :param P: population
        :param r: residual
        :param conf (dict) : configuration of original function
        :param stop_crit: stopping criterion
        :return:  P_post: new population,
                  fit_post: new fit,
                  r_post: new residual
        """
        if len(P.shape) < 2:
            P = P[np.newaxis, :]
        if isinstance(r, (int, float)):
            r = np.array([[r]])
        elif len(r.shape) < 2:
            r = r[np.newaxis, :]
        N, nParams = P.shape
        fit_post = np.zeros(N)
        P_post = np.zeros((N, nParams))
        r_post = np.zeros_like(r)

        for i in range(N):
            if self.verbose > 0: print(f' {i+1}/{N}')
            fit_post[i], P_post[i], r_post[i] = self.gauss_newton_slow(self.op,
                                                                       P[i],
                                                                       r[i],
                                                                       conf['gL'],
                                                                       conf['gU'],
                                                                       conf['gT'],
                                                                       conf['gLoop'],
                                                                       conf['gTol'],
                                                                       conf['LR'],
                                                                       conf['UR'],
                                                                       stop_crit)
        return P_post, fit_post, r_post

    def gauss_newton_slow(self, op, Para_E_test, r_test, reg0, reg1, steps, loop, tol, LR, UR, fit_crit):
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
            # print(f"[{j}/{loop}] ", end="")
            J = self.NMM_diff_A_lfm(Para_E_test, r_test)
            Para_E_new_group = self.multi_lavenberg_regularization(steps, reg0, reg1, Para_E_test, J, r_test, LR, UR)

            fit_grp, error_grp, hout_group = self.evaluation(Para_E_new_group, self.reference)
            Para_E_new, fit_new, r_new = self.selection_best(Para_E_new_group, error_grp, hout_group, 1, op)
            r_new = r_new.flatten()
            # this was changed
            # TODO check against original: self.selection_best(Para_E_new_group, fit_grp, error_grp, 1, op)

            r_test = r_new
            Para_E_test = Para_E_new.flatten()

            # print(fit_new)
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

    def NMM_diff_A_lfm(self, parameter, h_output):
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
        p_shape = parameter.shape[0]
        if isinstance(h_output, (int, float)):
            h_shape = 1
        else:
            # TODO: this only works if h_output is a scalar or 1D, make the more general if more than one output dim is
            #  needed
            h_shape = h_output.shape[-1]


        j = np.zeros((h_shape, p_shape))

        for i in range(p_shape):
            parameter_update = parameter.copy()
            parameter_update[i] = parameter_pert[i]
            h_output_new = self.function_call(parameter_update)
            j[:, i] = (h_output_new - h_output) / h

        j[np.isnan(j)] = 0
        j[np.isinf(j)] = 0

        return j

    def mutation_single(self, solution, LR, UR):
        """
        Single-parameter mutation on the best solution.

        Parameters:
            solution (np.ndarray): 1D array of best solution parameters.
            LR (float): Lower boundary for mutation.
            UR (float): Upper boundary for mutation.

        Returns:
            p (np.ndarray): 2D array of mutated solutions (nParams x nParams).
        """
        solution = np.array(solution)
        nParams = solution.size
        P = np.tile(solution, (nParams, 1))
        mutation_values = self.population(1, nParams, LR, UR).flatten()
        np.fill_diagonal(P, mutation_values)
        return P

    def selection_uniq(self, P1, E, p, r, op, LR, UR):
        """
        Selects a unique subset of the population based on fitness.

        Parameters:
            P1 (np.ndarray): Population matrix.
            E (np.ndarray): Fitness values.
            p (int): Desired population size.
            r (int): Number of top individuals to retain.
            op (float): Operation factor (e.g., -1 to invert fitness).
            LR (float): Lower bound for new individuals.
            UR (float): Upper bound for new individuals.

        Returns:
            P1_new (np.ndarray): Selected population.
            E_new (np.ndarray): Transformed fitness values.
        """
        E = op * E
        E_orig = E.copy()
        dim = P1.shape[1]

        # Remove inf and nan entries
        valid_mask = ~np.isinf(E) & ~np.isnan(E)
        E = E[valid_mask]
        P1 = P1[valid_mask]

        # Keep unique rows
        P1_unique, idx = np.unique(P1, axis=0, return_index=True)
        E = E[idx]
        P1 = P1_unique

        # Sort by fitness descending
        sorted_idx = np.argsort(E)
        P1 = P1[sorted_idx]
        E_sorted = E[sorted_idx]

        n_E = len(E_sorted)
        if n_E < p:
            n_new = p - n_E
            P1 = np.vstack([P1, self.population(n_new, dim, LR, UR)])
            E_sorted = np.concatenate([E_sorted, np.full(n_new, np.nan)])

        if n_E > p:
            P1_best = P1[:r]
            E_best = E_sorted[:r]
            P2 = P1[r:]
            E2 = E_sorted[r:]
            rand_idx = np.random.permutation(len(E2))
            P2 = P2[rand_idx][:p - r]
            E2 = E2[rand_idx][:p - r]
            P1 = np.vstack([P1_best, P2])
            E_sorted = np.concatenate([E_best, E2])

        P1_new = P1[:p] #flip added here
        E_new = op * E_sorted[:p]

        return P1_new, E_new

    def evaluation(self, X, y):
        """
        GA toolbox evaluation function that helps evaluating a simulation function 'func' over an array 'X' of values
        :param X: parameter values
        :param func: simulation function
        :param y: reference
        :return: fits: fit values (errors), h_outs: output values of evaluated functions
        """
        x_shape = X.shape[0]
        errors = np.zeros(x_shape)
        h_outs = np.zeros((self.t_shape, x_shape))
        print('houts', h_outs.shape)
        print('tshape', self.t_shape)
        print('y shape' , y.shape)
        fits = np.zeros(x_shape)

        for j in range(x_shape):
            P = X[j]
            start_time = time.time()
            h_out = self.function_call(P)
            error = nrmse(h_out, y)
            print('error', error)
            fit = np.sum(error**2)
            end_time = time.time()
            sim_time = end_time-start_time
            sim_time_float, sim_time_str = t_format(sim_time)
            if self.verbose > 0: print(f' simulation time: {sim_time_float:.3f}{sim_time_str} --> {j+1}/{x_shape}')
            fits[j] = fit
            errors[j] = error
            h_outs[:, j] = h_out
        h_outs = h_outs.T
        return fits, errors, h_outs

    def multi_lavenberg_regularization(self, n, reg0, reg1, Para_E, J, h_output, LR, UR):
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
        Y = np.zeros((n, Para_E.shape[0]))
        reg = 10 ** np.linspace(reg0, reg1, n)
        if isinstance(h_output, (int, float)):
            h_output = np.array([h_output])

        for i in range(reg.shape[0]):
            try:
                D = np.linalg.pinv(J.T @ J + reg[i] * np.eye(Para_E.shape[0]))
            except:
                # TODO: try if this error is error important and find out what exception is necessary
                return Y  # return current Y if inversion fails
            if isinstance(h_output, (int, float)):
                d = -D @ J.T * h_output
            else:
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

    def function_call(self, parameters):
        """
        Function that evaluates a simulation class by wrapping the parameters into a dict with linked parameter names
        :param parameters: (np.ndarray) Array of parameter values that need to be evaluated
        """
        keywords = self.parameters
        m = parameters.shape[0]
        # h_out = np.zeros((m, self.t_shape))
        for i in range(m):
            # for j in range(self.n_param):
            #     keywords[self.model_parameters[j]] = parameters[i, j]
            #
            # if self.simulation_class != None:
            #     self.simulation_class.__init__(parameters=keywords)
            #     if self.x_out != None:
            #         self.simulation_class.simulate()
            #         h_out[i] = eval(f'self.simulation_class.{self.x_out}')
            #     else:
            #         h_out[i] = self.simulation_class.simulate()
            keywords[self.model_parameters[i]] = parameters[i]

        if self.simulation_class != None:
            self.simulation_class.__init__(parameters=keywords)
            if self.x_out != None:
                self.simulation_class.simulate()
                h_out = eval(f'self.simulation_class.{self.x_out}')
            else:
                h_out = self.simulation_class.simulate()
        else:
            h_out = self.simulate(keywords)
        if isinstance(h_out, (int, float)):
            h_out = np.array([h_out, np.newaxis])
        return h_out