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


    def simulate(self, dv=None, dt=None, T=None):

        if dv is not None:
            self.dv = dv
        if dt is not None:
            self.dt = dt
        if T is not None:
            self.T = T

    def get_diffusion_coeffs(self):

        # synapse parameters
        mu_exc_e = 0.008
        mu_exc_i = 0.027
        mu_inh_e = 0.020
        mu_inh_i = 0.066

        coeff_var = 0.5
        var_exc_e = (coeff_var * mu_exc_e) ** 2
        var_exc_i = (coeff_var * mu_exc_i) ** 2
        var_inh_e = (coeff_var * mu_inh_e) ** 2
        var_inh_i = (coeff_var * mu_inh_i) ** 2

        a_exc_e = mu_exc_e ** 2 / var_exc_e
        a_exc_i = mu_exc_i ** 2 / var_exc_i
        a_inh_e = mu_inh_e ** 2 / var_inh_e
        a_inh_i = mu_inh_i ** 2 / var_inh_i

        scale_exc_e = var_exc_e / mu_exc_e
        scale_exc_i = var_exc_i / mu_exc_i
        scale_inh_e = var_inh_e / mu_inh_e
        scale_inh_i = var_inh_i / mu_inh_i

        # conductance jump distributions
        gamma_exc_e = scipy.stats.gamma(a=a_exc_e, loc=0, scale=scale_exc_e)
        # TODO: replace
        #gamma_exc_e = gamma(a=coeff_var**(-2), loc=0, scale=scale_exc_e)
        gamma_exc_i = scipy.stats.gamma(a=a_exc_i, loc=0, scale=scale_exc_i)
        gamma_inh_e = scipy.stats.gamma(a=a_inh_e, loc=0, scale=scale_inh_e)
        gamma_inh_i = scipy.stats.gamma(a=a_inh_i, loc=0, scale=scale_inh_i)

        # x = np.linspace(0, 0.1, 100)
        # Fe = gamma_exc_e.sf(x=x)
        # Fi = gamma_exc_i.sf(x=x)
        # plt.plot(x, Fe)
        # plt.plot(x, Fi)

        dv_fine = 0.01
        v_fine = np.arange(u_inh, u_thr + dv_fine, dv_fine)

        # coefficients for diffusion equation
        c1e_exc_fine = np.zeros(len(v_fine))
        c2e_exc_fine = np.zeros(len(v_fine))
        c1i_exc_fine = np.zeros(len(v_fine))
        c2i_exc_fine = np.zeros(len(v_fine))

        c1e_inh_fine = np.zeros(len(v_fine))
        c2e_inh_fine = np.zeros(len(v_fine))
        c1i_inh_fine = np.zeros(len(v_fine))
        c2i_inh_fine = np.zeros(len(v_fine))

        for i, v_ in enumerate(v_fine):
            vpe = np.arange(u_inh, v_ + dv_fine, dv_fine)
            int_exc_c1e = gamma_exc_e.sf((v_ - vpe) / (u_exc - vpe))
            int_exc_c2e = int_exc_c1e * (v_ - vpe)
            int_inh_c1e = gamma_inh_e.sf((v_ - vpe) / (u_exc - vpe))
            int_inh_c2e = int_inh_c1e * (v_ - vpe)

            c1e_exc_fine[i] = np.trapz(x=vpe, y=int_exc_c1e)
            c2e_exc_fine[i] = np.trapz(x=vpe, y=int_exc_c2e)
            c1e_inh_fine[i] = np.trapz(x=vpe, y=int_inh_c1e)
            c2e_inh_fine[i] = np.trapz(x=vpe, y=int_inh_c2e)

            if i > 0:
                vpi = np.arange(v_, u_thr + dv_fine, dv_fine)
                int_exc_c1i = gamma_exc_i.sf((v_ - vpi) / (u_inh - vpi))
                int_exc_c2i = int_exc_c1i * (vpi - v_)
                int_inh_c1i = gamma_inh_i.sf((v_ - vpi) / (u_inh - vpi))
                int_inh_c2i = int_inh_c1i * (vpi - v_)

                c1i_exc_fine[i] = np.trapz(x=vpi, y=int_exc_c1i)
                c2i_exc_fine[i] = np.trapz(x=vpi, y=int_exc_c2i)
                c1i_inh_fine[i] = np.trapz(x=vpi, y=int_inh_c1i)
                c2i_inh_fine[i] = np.trapz(x=vpi, y=int_inh_c2i)
