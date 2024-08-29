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