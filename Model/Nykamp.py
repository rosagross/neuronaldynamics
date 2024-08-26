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

        self.get_diffusion_coeffs(i=0)

    def get_diffusion_coeffs(self, k):

        # TODO: make these matrix operations, get all diff coeffs at once?

        # synapse parameters
        mu_ee = self.mu_gamma[k, 0]
        mu_ei = self.mu_gamma[k, 1]
        mu_ie = self.mu_gamma[k, 2]
        mu_ii = self.mu_gamma[k, 3]

        coeff_var = self.var_coeff_gamma[k]
        var_ee = (coeff_var * mu_ee) ** 2
        var_ei = (coeff_var * mu_ei) ** 2
        var_ie = (coeff_var * mu_ie) ** 2
        var_ii = (coeff_var * mu_ii) ** 2

        a_exc_e = mu_ee ** 2 / var_ee
        a_exc_i = mu_ei ** 2 / var_ei
        a_inh_e = mu_ie ** 2 / var_ie
        a_inh_i = mu_ii ** 2 / var_ii

        scale_exc_e = var_ee / mu_ee
        scale_exc_i = var_ei / mu_ei
        scale_inh_e = var_ie / mu_ie
        scale_inh_i = var_ii / mu_ii

        # conductance jump distributions
        gamma_ee = scipy.stats.gamma(a=a_exc_e, loc=0, scale=scale_exc_e)
        #gamma_exc_e = gamma(a=coeff_var**(-2), loc=0, scale=scale_exc_e)
        gamma_ei = scipy.stats.gamma(a=a_exc_i, loc=0, scale=scale_exc_i)
        gamma_ie = scipy.stats.gamma(a=a_inh_e, loc=0, scale=scale_inh_e)
        gamma_ii = scipy.stats.gamma(a=a_inh_i, loc=0, scale=scale_inh_i)

        # leave for now v_fine = np.arange(self.u_inh, self.u_thr + self.dv_fine, self.dv_fine)
        v = np.arange(self.u_inh, self.u_thr + self.dv, self.dv)

        c1ee = np.zeros(len(v))
        c2ee = np.zeros(len(v))
        c1ei = np.zeros(len(v))
        c2ei = np.zeros(len(v))
        c1ie = np.zeros(len(v))
        c2ie = np.zeros(len(v))
        c1ii = np.zeros(len(v))
        c2ii = np.zeros(len(v))

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
            # TODO: put them into a useful self format
