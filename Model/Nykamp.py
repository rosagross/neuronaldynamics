import time
import scipy
import numpy as np
import matplotlib.pyplot as plt
import numba

import scipy
from tqdm import tqdm
from scipy.stats import gamma
from scipy.stats import norm
import h5py
import matplotlib
matplotlib.use('TkAgg')

class Nykamp_Model():
    """
    Class for the population model of Nykamp and Tranchina
    Probability flux over time is used to model synaptic membrane and firing rates of neuron
    populations
    """

    def __init__(self, parameters):

        self.connectivity_matrix = np.ones([2, 2])
        self.tau_alpha = 1/3
        self.n_alpha = 9

        self.dt = 0.1
        self.dv = 0.01
        self.T = 100

        if 'connectivity_matrix' in parameters:
            self.connectivity_matrix = parameters['connectivity_matrix']
        if 'u_rest' in parameters:
            self.u_rest = parameters['u_rest']
        if 'u_reset' in parameters:
            self.u_reset = parameters['u_reset']
        else:
            self.u_reset = parameters['u_rest']
        if 'u_thr' in parameters:
            self.u_thr = parameters['u_thr']
        if 'u_exc' in parameters:
            self.u_exc = parameters['exc']
        if 'u_inh' in parameters:
            self.u_inh = parameters['u_inh']
        if 'tau_mem' in parameters:
            self.tau_mem = parameters['tau_mem']
        if 'tau_ref' in parameters:
            self.tau_ref = parameters['tau_ref']
        if 'mu_gamma' in parameters:
            self.mu_gamma = parameters['mu_gamma']
        if 'var_coeff_gamma' in parameters:
            self.var_coeff_gamma = parameters['var_coeff_gamma']
        if 'tau_alpha' in parameters:
            self.tau_alpha = parameters['tau_alpha']
        if 'n_alpha' in parameters:
            self.n_alpha = parameters['n_alpha']
        if 'input_function' in parameters:
            self.input_function = parameters['input_function']
        if 'input_function_type' in parameters:
            self.input_function_type = parameters['input_function_type']
        if 'input_function_idx' in parameters:
            self.input_function_idx = parameters['input_function_idx']
        if 'population_type' in parameters:
            self.population_type = parameters['population_type']

        self.input = np.zeros(len(self.population_type))

        n_coeff = 4 * len(self.population_type)
        self.diffusion_coefficients = np.zeros(n_coeff)
        self.diffusion_coefficients_dv = np.zeros(n_coeff)
        self.dFdv = np.zeros(int(n_coeff/2))

    def simulate(self, dv=None, dt=None, T=None, dv_fine=None):

        if dv is not None:
            self.dv = dv
        if dt is not None:
            self.dt = dt
        if T is not None:
            self.T = T
        if dv_fine is None:
            self.dv_fine = dv
        else:
            self.dv_fine = dv_fine

        self.t = np.arange(0, T, dt)
        self.v = np.arange(self.u_inh, self.u_thr + dv, dv)

        # set up input_function over time
        self.input[self.input_function_idx] = self.input_function(t=self.t)

        # calculate alpha kernel
        self.get_alpha_kernel()

        self.c1ee, self.c2ee, self.c1ei, self.c2ei,self.c1ie, self.c2ie, self.c1ii,\
            self.c2ii = self.get_diffusion_coeffs_classic()

        # calculate diffusion coefficients
        self.get_diffusion_coeffs()
        self.diffusion_coefficients_dv = np.gradient(self.diffusion_coefficients, self.dv, axis=-1)

        self.c1ee_v = self.diffusion_coefficients_dv[0, 0]
        self.c2ee_v = self.diffusion_coefficients_dv[0, 1]
        self.c1ie_v = self.diffusion_coefficients_dv[0, 2]
        self.c2ie_v = self.diffusion_coefficients_dv[0, 3]

        self.c1ei_v = self.diffusion_coefficients_dv[1, 0]
        self.c2ei_v = self.diffusion_coefficients_dv[1, 1]
        self.c1ii_v = self.diffusion_coefficients_dv[1, 2]
        self.c2ii_v = self.diffusion_coefficients_dv[1, 3]

        # self.c1ie_v = np.gradient(self.c1ie, self.dv)
        # self.c2ie_v = np.gradient(self.c2ie, self.dv)
        # self.c1ii_v = np.gradient(self.c1ii, self.dv)
        # self.c2ii_v = np.gradient(self.c2ii, self.dv)
        #
        # self.c1ee_v = np.gradient(self.c1ee, self.dv)
        # self.c2ee_v = np.gradient(self.c2ee, self.dv)
        # self.c1ei_v = np.gradient(self.c1ei, self.dv)
        # self.c2ei_v = np.gradient(self.c2ei, self.dv)

        for i in range(len(self.population_type)):
            # initialize arrays
            #TODO: check how many of those are replications and eventually use a single array for all of these
            if self.population_type[i] == 'exc':

                rho_exc = np.zeros((len(self.v), len(self.t))) # probability density of discontinuous membrane potential
                rho_exc_delta = np.zeros(len(self.t)) # probability density of membrane potential
                ref_exc_delta_idx = int(self.tau_ref / self.dt)  # number of time steps of refractory delay
                v_reset_idx = np.where(np.isclose(self.v, self.u_reset))[0][0]  # index of reset potential in array
                r_exc = np.zeros(len(self.t))  # output firing rate
                r_exc_delayed = np.zeros(len(self.t) + ref_exc_delta_idx)  # delayed output firing rate
                v_in_ee = np.zeros(len(self.t))
                v_in_ei = np.zeros(len(self.t))


            elif self.population_type[i] == 'inh':
                rho_inh = np.zeros((len(self.v), len(self.t)))  # probability density of membrane potential
                rho_inh_delta = np.zeros(len(self.t))  # probability density of discontinuous membrane potential
                ref_inh_delta_idx = int(self.tau_ref / self.dt)  # number of time steps of refractory delay
                v_reset_idx = np.where(np.isclose(self.v, self.u_reset))[0][0]  # index of reset potential in array
                r_inh = np.zeros(len(self.t))  # output firing rate
                r_inh_delayed = np.zeros(len(self.t) + ref_inh_delta_idx)  # delayed output firing rate
                v_in_ie = np.zeros(len(self.t))
                v_in_ii = np.zeros(len(self.t))

        # Determine population dynamics (diffusion approximation)
        for i, t_ in enumerate(tqdm(self.t[:-1])):
            # for i, t_ in enumerate(t[:-1]):

            # excitatory population
            # ================================================================================================================
            # input to excitatory population
            # TODO: this needs to be done for each population at the start, best as a vector
            if i > 0:
                r_exc_conv = np.convolve(r_exc[:(i + 1)], self.alpha)[-len(self.alpha)] * self.dt
                r_inh_conv = np.convolve(r_inh[:(i + 1)], self.alpha)[-len(self.alpha)] * self.dt
            else:
                r_exc_conv = 0
                r_inh_conv = 0

            # TODO: think about a flattened version of all idxs here for iterating over all connections
            #  for each time step

            v_in_ee[i] = self.connectivity_matrix[0, 0] * r_exc_conv + self.input[0]
            v_in_ei[i] = self.connectivity_matrix[0, 1] * r_inh_conv + self.input[1]

            # coefficients for finite difference matrices
            # c1, c2 are over all v steps and i is a time step
            f0_exc = dt / 2 * (1 / self.tau_mem[0] - v_in_ee[i] * self.c1ee_v + v_in_ei[i] * self.c1ie_v)

            f1_exc = dt / (4 * dv) * (
                        (self.v - self.u_rest) / self.tau_mem[0] + v_in_ee[i] * (-self.c1ee + self.c2ee_v)
                        + v_in_ei[i] * (self.c1ie + self.c2ie_v))
            f2_exc = dt / (2 * dv ** 2) * (v_in_ee[i] * self.c2ee + v_in_ei[i] * self.c2ie)

            # LHS matrix (t+dt)
            A_exc = np.diag(1 + 2 * f2_exc - f0_exc) + np.diagflat((-f2_exc - f1_exc)[:-1], 1) + np.diagflat(
                (f1_exc - f2_exc)[1:], -1)
            A_exc[0, 1] = -2 * f1_exc[1]
            A_exc[-1, -2] = 2 * f1_exc[-2]

            # RHS matrix (t)
            B_exc = np.diag(1 - 2 * f2_exc + f0_exc) + np.diagflat((f2_exc + f1_exc)[:-1], 1) + np.diagflat(
                (f2_exc - f1_exc)[1:], -1)
            B_exc[0, 1] = 2 * f1_exc[1]
            B_exc[-1, -2] = -2 * f1_exc[-2]

            # contribution to drho/dt from rho_delta at u_res
            g_exc = rho_exc_delta[i] * (-v_in_exc_exc[i] * dFe_exc_delta_dv + v_in_exc_inh[i] * dFi_exc_delta_dv)

            # calculate firing rate
            r_exc[i] = v_in_exc_exc[i] * (
                        c2e_exc[-1] * rho_exc[-2, i] / dv + gamma_ee.sf((u_thr - u_res) / (u_exc - u_res)) *
                        rho_exc_delta[i])
            if r_exc[i] < 0:
                # print(f"WARNING: r_exc < 0 ! (r_exc = {r_exc[i]}) ... Setting r_exc to 0")
                r_exc[i] = 0
            r_exc_delayed[i + ref_exc_delta_idx] = r_exc[i]

            # update rho and rho_delta
            # rho_exc[:, i+1] = np.linalg.solve(A_exc, np.matmul(B_exc, rho_exc[:, i][:, np.newaxis]))[:, 0]
            # old overly complicated version
            rho_exc[:, i + 1] = np.linalg.solve(A_exc, np.matmul(B_exc, rho_exc[:, i]))
            rho_exc[:, i + 1] += dt * g_exc
            rho_exc_delta[i + 1] = rho_exc_delta[i] + dt * (
                        -(v_in_exc_exc[i] + v_in_exc_inh[i]) * rho_exc_delta[i] + r_exc_delayed[i])

            # inhibitory population
            # ================================================================================================================
            # input to inhibitory population
            v_in_inh_exc[i] = w_ei * r_exc_conv
            v_in_inh_inh[i] = w_ii * r_inh_conv

            # coefficients for finite difference matrices
            f0_inh = dt / 2 * (1 / tau_inh_membrane - v_in_inh_exc[i] * dc1e_inh_dv + v_in_inh_inh[i] * dc1i_inh_dv)
            f1_inh = dt / (4 * dv) * (
                        (v - u_res) / tau_inh_membrane + v_in_inh_exc[i] * (-c1e_inh + dc2e_inh_dv) + v_in_inh_inh[
                    i] * (c1i_inh + dc2i_inh_dv))
            f2_inh = dt / (2 * dv ** 2) * (v_in_inh_exc[i] * c2e_inh + v_in_inh_inh[i] * c2i_inh)

            # LHS matrix (t+dt)
            A_inh = np.diag(1 + 2 * f2_inh - f0_inh) + np.diagflat((-f2_inh - f1_inh)[:-1], 1) + np.diagflat(
                (f1_inh - f2_inh)[1:], -1)
            A_inh[0, 1] = -2 * f1_inh[1]
            A_inh[-1, -2] = 2 * f1_inh[-2]

            # RHS matrix (t)
            B_inh = np.diag(1 - 2 * f2_inh + f0_inh) + np.diagflat((f2_inh + f1_inh)[:-1], 1) + np.diagflat(
                (f2_inh - f1_inh)[1:], -1)
            B_inh[0, 1] = 2 * f1_inh[1]
            B_inh[-1, -2] = -2 * f1_inh[-2]

            # contribution to drho/dt from rho_delta at u_res
            g_inh = rho_inh_delta[i] * (-v_in_inh_exc[i] * dFe_inh_delta_dv + v_in_inh_inh[i] * dFi_inh_delta_dv)

            # calculate firing rate
            r_inh[i] = v_in_inh_exc[i] * (
                        c2e_inh[-1] * rho_inh[-2, i] / dv + gamma_inh_e.sf((u_thr - u_res) / (u_exc - u_res)) *
                        rho_inh_delta[i])
            if r_inh[i] < 0:
                # print(f"WARNING: r_inh < 0 ! (r_inh = {r_inh[i]}) ... Setting r_inh to 0")
                r_inh[i] = 0
            r_inh_delayed[i + ref_inh_delta_idx] = r_inh[i]

            # update rho and rho_delta
            rho_inh[:, i + 1] = np.linalg.solve(A_inh, np.matmul(B_inh, rho_inh[:, i]))
            rho_inh[:, i + 1] += dt * g_inh
            rho_inh_delta[i + 1] = rho_inh_delta[i] + dt * (
                        -(v_in_inh_exc[i] + v_in_inh_inh[i]) * rho_inh_delta[i] + r_inh_delayed[i])

        rho_plot_exc = rho_exc
        rho_plot_exc[v_reset_idx, :] = rho_exc[v_reset_idx, :] + rho_exc_delta[:]
        r_plot_exc = r_exc[:]

        rho_plot_inh = rho_inh
        rho_plot_inh[v_reset_idx, :] = rho_inh[v_reset_idx, :] + rho_inh_delta[:]
        r_plot_inh = r_inh[:]

        with h5py.File('test' + '.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=t)
            h5file.create_dataset('v', data=v)
            h5file.create_dataset('r_exc', data=r_plot_exc)
            h5file.create_dataset('r_inh', data=r_plot_inh)
            h5file.create_dataset('rho_plot_exc', data=rho_plot_exc)
            h5file.create_dataset('rho_plot_inh', data=rho_plot_inh)

    def get_diffusion_coeffs_classic(self):

        v = np.arange(self.u_inh, self.u_thr + self.dv, self.dv)

        # init arrays for diffusion coeffs
        c1ee = np.zeros(len(v))
        c2ee = np.zeros(len(v))
        c1ei = np.zeros(len(v))
        c2ei = np.zeros(len(v))

        # synapse parameters
        mu_ee = self.mu_gamma[0, 0]
        mu_ei = self.mu_gamma[0, 1]

        coeff_var = self.var_coeff_gamma[0]
        var_ee = (coeff_var * mu_ee) ** 2
        var_ei = (coeff_var * mu_ei) ** 2

        scale_ee = var_ee / mu_ee
        scale_ei = var_ei / mu_ei

        # conductance jump distributions
        gamma_ee = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ee)
        gamma_ei = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ei)

        # init arrays for diffusion coeffs
        c1ie = np.zeros(len(v))
        c2ie = np.zeros(len(v))
        c1ii = np.zeros(len(v))
        c2ii = np.zeros(len(v))

        # synapse parameters
        mu_ie = self.mu_gamma[1, 2]
        mu_ii = self.mu_gamma[1, 3]

        coeff_var = self.var_coeff_gamma[1]
        var_ie = (coeff_var * mu_ie) ** 2
        var_ii = (coeff_var * mu_ii) ** 2

        scale_ie = var_ie / mu_ie
        scale_ii = var_ii / mu_ii

        # conductance jump distributions
        gamma_ie = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ie)
        gamma_ii = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ii)

        for i, v_ in enumerate(v):
            vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
            int_c1ee = gamma_ee.sf((v_ - vpe) / (self.u_exc - vpe))
            int_c2ee = int_c1ee * (v_ - vpe)
            int_c1ei = gamma_ei.sf((v_ - vpe) / (self.u_exc - vpe))
            int_c2ei = int_c1ei * (v_ - vpe)

            c1ee[i] = np.trapz(x=vpe, y=int_c1ee)
            c2ee[i] = np.trapz(x=vpe, y=int_c2ee)
            c1ei[i] = np.trapz(x=vpe, y=int_c1ei)
            c2ei[i] = np.trapz(x=vpe, y=int_c2ei)

            if i > 0:
                vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                int_c1ie = gamma_ie.sf((v_ - vpi) / (self.u_inh - vpi))
                int_c2ie = int_c1ie * (vpi - v_)
                int_c1ii = gamma_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                int_c2ii = int_c1ii * (vpi - v_)

                c1ie[i] = np.trapz(x=vpi, y=int_c1ie)
                c2ie[i] = np.trapz(x=vpi, y=int_c2ie)
                c1ii[i] = np.trapz(x=vpi, y=int_c1ii)
                c2ii[i] = np.trapz(x=vpi, y=int_c2ii)

        c1ie[0] = c1ie[1]
        c2ie[0] = 0
        c1ie[0] = c1ie[1]
        c2ie[0] = 0


        return c1ee, c2ee, c1ei, c2ei, c1ie, c2ie, c1ii, c2ii

    def get_diffusion_coeffs(self, k=0):

        # TODO: make these matrix operations, get all diffusion coeffs at once?

        v = np.arange(self.u_inh, self.u_thr + self.dv, self.dv)

        if self.population_type[k] == 'exc':

            # init arrays for diffuson coeffs
            c1ee = np.zeros(len(v))
            c2ee = np.zeros(len(v))
            c1ei = np.zeros(len(v))
            c2ei = np.zeros(len(v))

            # synapse parameters
            mu_ee = self.mu_gamma[k, 0]
            mu_ei = self.mu_gamma[k, 1]

            coeff_var = self.var_coeff_gamma[k]
            var_ee = (coeff_var * mu_ee) ** 2
            var_ei = (coeff_var * mu_ei) ** 2

            scale_ee = var_ee / mu_ee
            scale_ei = var_ei / mu_ei

            # conductance jump distributions
            gamma_ee = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ee)
            gamma_ei = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ei)

            for i, v_ in enumerate(v):
                vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
                int_c1ee = gamma_ee.sf((v_ - vpe) / (self.u_exc - vpe))
                int_c2ee = int_c1ee * (v_ - vpe)
                int_c1ei = gamma_ei.sf((v_ - vpe) / (self.u_exc - vpe))
                int_c2ei = int_c1ei * (v_ - vpe)

                c1ee[i] = np.trapz(x=vpe, y=int_c1ee)
                c2ee[i] = np.trapz(x=vpe, y=int_c2ee)
                c1ei[i] = np.trapz(x=vpe, y=int_c1ei)
                c2ei[i] = np.trapz(x=vpe, y=int_c2ei)

            self.diffusion_coefficients[k, :] = c1ee, c2ee, c1ei, c2ei
            Fee_v = np.gradient(gamma_ee.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                    * np.heaviside(self.v - self.u_rest, 0.5)
            Fei_v = np.gradient(gamma_ei.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                    * np.heaviside(self.v - self.u_rest, 0.5)

            self.dFdv[k] = np.array([Fee_v, Fei_v])


        elif self.population_type[k] == 'inh':

            # init arrays for diffusion coeffs
            c1ie = np.zeros(len(v))
            c2ie = np.zeros(len(v))
            c1ii = np.zeros(len(v))
            c2ii = np.zeros(len(v))

            # synapse parameters
            mu_ie = self.mu_gamma[k, 2]
            mu_ii = self.mu_gamma[k, 3]

            coeff_var = self.var_coeff_gamma[k]
            var_ie = (coeff_var * mu_ie) ** 2
            var_ii = (coeff_var * mu_ii) ** 2

            scale_ie = var_ie / mu_ie
            scale_ii = var_ii / mu_ii

            # conductance jump distributions
            gamma_ie = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ie)
            gamma_ii = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ii)

            for i, v_ in enumerate(v):
                if i > 0:
                    vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                    int_c1ie = gamma_ie.sf((v_ - vpi) / (self.u_inh - vpi))
                    int_c2ie = int_c1ie * (vpi - v_)
                    int_c1ii = gamma_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                    int_c2ii = int_c1ii * (vpi - v_)

                    c1ie[i] = np.trapz(x=vpi, y=int_c1ie)
                    c2ie[i] = np.trapz(x=vpi, y=int_c2ie)
                    c1ii[i] = np.trapz(x=vpi, y=int_c1ii)
                    c2ii[i] = np.trapz(x=vpi, y=int_c2ii)

            c1ie[0] = c1ie[1]
            c2ie[0] = 0
            c1ie[0] = c1ie[1]
            c2ie[0] = 0

            self.diffusion_coefficients[k, :] = c1ie, c2ie, c1ii, c2ii


            Fie_v = np.gradient(gamma_ie.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                               * np.heaviside(self.u_rest - self.v, 0.5)
            Fii_v = np.gradient(gamma_ii.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                               * np.heaviside(self.u_rest - self.v, 0.5)
            self.dFdv[k] = np.array([Fie_v, Fii_v])

        else:
            raise NotImplementedError('population types must be "exc" or "inh"!')

    def get_alpha_kernel(self):
        self.t_alpha = self.t[self.t < 10]
        self.alpha = np.exp(-self.t_alpha/self.tau_alpha) / (self.tau_alpha * scipy.special.factorial(self.n_alpha-1)) *\
                (self.t_alpha/self.tau_alpha)**(self.n_alpha-1)
        self.alpha = self.alpha/np.trapz(self.alpha, dx=self.dt)