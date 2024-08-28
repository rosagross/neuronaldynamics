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

        self.c1ee, self.c2ee, self.c1ei, self.c2ei,self.c1ie, self.c2ie, self.c1ii,\
            self.c2ii = self.get_diffusion_coeffs_classic()

        self.c1ie_v = np.gradient(self.c1ie, self.dv)
        self.c2ie_v = np.gradient(self.c2ie, self.dv)
        self.c1ii_v = np.gradient(self.c1ie, self.dv)
        self.c2ii_v = np.gradient(self.c2ie, self.dv)

        self.c1ee_v = np.gradient(self.c1ee, self.dv)
        self.c2ee_v = np.gradient(self.c2ee, self.dv)
        self.c1ei_v = np.gradient(self.c1ee, self.dv)
        self.c2ei_v = np.gradient(self.c2ee, self.dv)
    def get_diffusion_coeffs_classic(self):

        v = np.arange(self.u_inh, self.u_thr + self.dv, self.dv)

        # init arrays for diffuson coeffs
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

        scale_exc_e = var_ee / mu_ee
        scale_exc_i = var_ei / mu_ei

        # conductance jump distributions
        gamma_ee = gamma(a=coeff_var ** (-2), loc=0, scale=scale_exc_e)
        gamma_ei = gamma(a=coeff_var ** (-2), loc=0, scale=scale_exc_i)

        # init arrays for diffuson coeffs
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

        scale_inh_e = var_ie / mu_ie
        scale_inh_i = var_ii / mu_ii

        # conductance jump distributions
        gamma_ie = gamma(a=coeff_var ** (-2), loc=0, scale=scale_inh_e)
        gamma_ii = gamma(a=coeff_var ** (-2), loc=0, scale=scale_inh_i)

        for i, v_ in enumerate(v):
            vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
            int_exc_c1e = gamma_ee.sf((v_ - vpe) / (self.u_exc - vpe))
            int_exc_c2e = int_exc_c1e * (v_ - vpe)
            int_inh_c1e = gamma_ie.sf((v_ - vpe) / (self.u_exc - vpe))
            int_inh_c2e = int_inh_c1e * (v_ - vpe)

            c1ee[i] = np.trapz(x=vpe, y=int_exc_c1e)
            c2ee[i] = np.trapz(x=vpe, y=int_exc_c2e)
            c1ei[i] = np.trapz(x=vpe, y=int_inh_c1e)
            c2ei[i] = np.trapz(x=vpe, y=int_inh_c2e)

            if i > 0:
                vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                int_exc_c1i = gamma_ei.sf((v_ - vpi) / (self.u_inh - vpi))
                int_exc_c2i = int_exc_c1i * (vpi - v_)
                int_inh_c1i = gamma_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                int_inh_c2i = int_inh_c1i * (vpi - v_)

                c1ie[i] = np.trapz(x=vpi, y=int_exc_c1i)
                c2ie[i] = np.trapz(x=vpi, y=int_exc_c2i)
                c1ii[i] = np.trapz(x=vpi, y=int_inh_c1i)
                c2ii[i] = np.trapz(x=vpi, y=int_inh_c2i)

        c1ie[0] = c1ie[1]
        c2ie[0] = 0
        c1ie[0] = c1ie[1]
        c2ie[0] = 0

        return c1ee, c2ee, c1ei, c2ei, c1ie, c2ie, c1ii, c2ii

    def get_diffusion_coeffs(self, k=0):

        # TODO: make these matrix operations, get all diff coeffs at once?

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

            scale_exc_e = var_ee / mu_ee
            scale_exc_i = var_ei / mu_ei

            # conductance jump distributions
            gamma_ee = gamma(a=coeff_var ** (-2), loc=0, scale=scale_exc_e)
            gamma_ei = gamma(a=coeff_var ** (-2), loc=0, scale=scale_exc_i)

            for i, v_ in enumerate(v):
                vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
                int_exc_c1e = gamma_ee.sf((v_ - vpe) / (self.u_exc - vpe))
                int_exc_c2e = int_exc_c1e * (v_ - vpe)
                # here gamma ei was used
                int_inh_c1e = gamma_ie.sf((v_ - vpe) / (self.u_exc - vpe))
                int_inh_c2e = int_inh_c1e * (v_ - vpe)

                c1ee[i] = np.trapz(x=vpe, y=int_exc_c1e)
                c2ee[i] = np.trapz(x=vpe, y=int_exc_c2e)
                c1ei[i] = np.trapz(x=vpe, y=int_inh_c1e)
                c2ei[i] = np.trapz(x=vpe, y=int_inh_c2e)

                if i > 0:
                    vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                    int_exc_c1i = gamma_ei.sf((v_ - vpi) / (self.u_inh - vpi))
                    int_exc_c2i = int_exc_c1i * (vpi - v_)
                    int_inh_c1i = gamma_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                    int_inh_c2i = int_inh_c1i * (vpi - v_)

            return c1ee, c2ee, c1ei, c2ei

        elif self.population_type[k] == 'inh':

            # init arrays for diffuson coeffs
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

            scale_inh_e = var_ie / mu_ie
            scale_inh_i = var_ii / mu_ii

            # conductance jump distributions
            gamma_ie = gamma(a=coeff_var ** (-2), loc=0, scale=scale_inh_e)
            gamma_ii = gamma(a=coeff_var ** (-2), loc=0, scale=scale_inh_i)

            for i, v_ in enumerate(v):
                vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
                int_exc_c1e = gamma_ee.sf((v_ - vpe) / (self.u_exc - vpe))
                int_exc_c2e = int_exc_c1e * (v_ - vpe)
                int_inh_c1e = gamma_ie.sf((v_ - vpe) / (self.u_exc - vpe))
                int_inh_c2e = int_inh_c1e * (v_ - vpe)

                if i > 0:
                    vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                    int_exc_c1i = gamma_ei.sf((v_ - vpi) / (self.u_inh - vpi))
                    int_exc_c2i = int_exc_c1i * (vpi - v_)
                    int_inh_c1i = gamma_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                    int_inh_c2i = int_inh_c1i * (vpi - v_)

                    c1ie[i] = np.trapz(x=vpi, y=int_exc_c1i)
                    c2ie[i] = np.trapz(x=vpi, y=int_exc_c2i)
                    c1ii[i] = np.trapz(x=vpi, y=int_inh_c1i)
                    c2ii[i] = np.trapz(x=vpi, y=int_inh_c2i)

            c1ie[0] = c1ie[1]
            c2ie[0] = 0
            c1ie[0] = c1ie[1]
            c2ie[0] = 0
            return c1ie, c2ie, c1ii, c2ii

        else:
            raise NotImplementedError('population types must be "exc" or "inh"!')


