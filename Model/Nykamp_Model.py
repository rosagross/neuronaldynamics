import time
import scipy
import numpy as np
import matplotlib.pyplot as plt

import scipy
from tqdm import tqdm
from scipy.stats import gamma
from scipy.stats import norm
import os
import h5py
import matplotlib
matplotlib.use('TkAgg')

class Nykamp_Model():
    """
    Class for the population model of Nykamp and Tranchina
    Probability flux over time is used to model synaptic membrane and firing rates of neuron
    populations
    """

    def __init__(self, parameters, name='Nykamp'):

        self.connectivity_matrix = np.ones([2, 2])
        self.tau_alpha = 1/3
        self.n_alpha = 9

        self.dt = 0.1
        self.dv = 0.01
        self.T = 100

        self.name = name

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
            self.u_exc = parameters['u_exc']
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

        self.n_populations = len(self.population_type)


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

        self.input = np.zeros([self.n_populations, self.n_populations, self.t.shape[0]])
        # set up input_function over time
        self.input[self.input_function_idx[0], self.input_function_idx[1]] = self.input_function(t=self.t)

        # calculate alpha kernel
        self.get_alpha_kernel()

        # self.c1ee, self.c2ee, self.c1ei, self.c2ei,self.c1ie, self.c2ie, self.c1ii,\
        #     self.c2ii = self.get_diffusion_coeffs_classic()

        # calculate diffusion coefficients
        self.get_diffusion_coeffs()
        self.diffusion_coefficients_dv = np.gradient(self.diffusion_coefficients, self.dv, axis=-1)

        self.c1ee = self.diffusion_coefficients[0]
        self.c2ee = self.diffusion_coefficients[1]
        self.c1ei = self.diffusion_coefficients[2]
        self.c2ei = self.diffusion_coefficients[3]

        self.c1ie = self.diffusion_coefficients[4]
        self.c2ie = self.diffusion_coefficients[5]
        self.c1ii = self.diffusion_coefficients[6]
        self.c2ii = self.diffusion_coefficients[7]

        self.c1ee_v = self.diffusion_coefficients_dv[0]
        self.c2ee_v = self.diffusion_coefficients_dv[1]
        self.c1ei_v = self.diffusion_coefficients_dv[2]
        self.c2ei_v = self.diffusion_coefficients_dv[3]

        self.c1ie_v = self.diffusion_coefficients_dv[4]
        self.c2ie_v = self.diffusion_coefficients_dv[5]
        self.c1ii_v = self.diffusion_coefficients_dv[6]
        self.c2ii_v = self.diffusion_coefficients_dv[7]

        for i in range(len(self.population_type)):
            # initialize arrays
            v_reset_idx = np.where(np.isclose(self.v, self.u_reset))[0][0]  # index of reset potential in array
            self.v_reset_idx = v_reset_idx

            #TODO: check how many of those are replications and eventually use a single array for all of these
            if self.population_type[i] == 'exc':

                rho_exc = np.zeros((len(self.v), len(self.t))) # probability density of discontinuous membrane potential
                rho_exc_delta = np.zeros(len(self.t)) # probability density of membrane potential
                ref_exc_delta_idx = int(self.tau_ref[0] / self.dt)  # number of time steps of refractory delay
                r_exc = np.zeros(len(self.t))  # output firing rate
                r_exc_delayed = np.zeros(len(self.t) + ref_exc_delta_idx)  # delayed output firing rate
                v_in_ee = np.zeros(len(self.t))
                v_in_ei = np.zeros(len(self.t))

                rho_exc[:, 0] = scipy.stats.norm.pdf(self.v, self.u_rest, 1)
                rho_exc[0, 0] = 0
                rho_exc[-1, 0] = 0


            elif self.population_type[i] == 'inh':
                rho_inh = np.zeros((len(self.v), len(self.t)))  # probability density of membrane potential
                rho_inh_delta = np.zeros(len(self.t))  # probability density of discontinuous membrane potential
                ref_inh_delta_idx = int(self.tau_ref[1] / self.dt)  # number of time steps of refractory delay
                r_inh = np.zeros(len(self.t))  # output firing rate
                r_inh_delayed = np.zeros(len(self.t) + ref_inh_delta_idx)  # delayed output firing rate
                v_in_ie = np.zeros(len(self.t))
                v_in_ii = np.zeros(len(self.t))

                # initialize rho with a gaussian distribution around the resting potential
                rho_inh[:, 0] = scipy.stats.norm.pdf(self.v, self.u_rest, 1)
                rho_inh[0, 0] = 0
                rho_inh[-1, 0] = 0

            Fee_v = self.dFdv[0]
            Fei_v = self.dFdv[1]
            Fie_v = self.dFdv[2]
            Fii_v = self.dFdv[3]

        # Determine population dynamics (diffusion approximation)
        for i, t_ in enumerate(tqdm(self.t[:-1])):
            # for i, t_ in enumerate(t[:-1]):

            # excitatory population
            # ================================================================================================================
            # input to excitatory population
            if i > 0:
                r_exc_conv = np.convolve(r_exc[:(i + 1)], self.alpha)[-len(self.alpha)] * self.dt
                r_inh_conv = np.convolve(r_inh[:(i + 1)], self.alpha)[-len(self.alpha)] * self.dt
            else:
                r_exc_conv = 0
                r_inh_conv = 0

            v_in_ee[i] = self.connectivity_matrix[0, 0] * r_exc_conv + self.input[0, 0][i]
            v_in_ei[i] = self.connectivity_matrix[0, 1] * r_inh_conv + self.input[0, 1][i]
            v_in_ie[i] = self.connectivity_matrix[1, 0] * r_exc_conv + self.input[1, 0][i]
            v_in_ii[i] = self.connectivity_matrix[1, 1] * r_inh_conv + self.input[1, 1][i]

            # v_in_ee[i] = self.input[0, 0][i] + self.connectivity_matrix[0, 0] * r_exc_conv
            # v_in_ei[i] = self.connectivity_matrix[0, 1] * r_inh_conv

            # coefficients for finite difference matrices
            # c1, c2 are over all v steps and i is a time step

            f0_exc = self.dt / 2 * (1 / self.tau_mem[0] - v_in_ee[i] * self.c1ee_v + v_in_ei[i] * self.c1ei_v)

            f1_exc = self.dt / (4 * self.dv) * (
                    (self.v - self.u_rest) / self.tau_mem[0] + v_in_ee[i] * (-self.c1ee + self.c2ee_v) + v_in_ei[i] * (
                    self.c1ei + self.c2ei_v))
            f2_exc = self.dt / (2 * self.dv ** 2) * (v_in_ee[i] * self.c2ee + v_in_ei[i] * self.c2ei)


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
            g_exc = rho_exc_delta[i] * (-v_in_ee[i] * Fee_v + v_in_ei[i] * Fei_v)

            # calculate firing rate
            r_exc[i] = v_in_ee[i] * (self.c2ee[-1] * rho_exc[-2, i] / self.dv + self.gamma_ee.sf(
                (self.u_thr - self.u_rest) / (self.u_exc - self.u_rest)) * rho_exc_delta[i])
            if r_exc[i] < 0:
                r_exc[i] = 0
            r_exc_delayed[i + ref_exc_delta_idx] = r_exc[i]

            rho_exc[:, i + 1] = np.linalg.solve(A_exc, np.matmul(B_exc, rho_exc[:, i]))
            rho_exc[:, i + 1] += self.dt * g_exc
            rho_exc_delta[i + 1] = rho_exc_delta[i] + self.dt * (
                        -(v_in_ee[i] + v_in_ei[i]) * rho_exc_delta[i] + r_exc_delayed[i])

            # inhibitory population
            # ================================================================================================================
            # input to inhibitory population


            # coefficients for finite difference matrices
            f0_inh = self.dt / 2 * (1 / self.tau_mem[1] - v_in_ie[i] * self.c1ie_v + v_in_ii[i] * self.c1ii_v)
            f1_inh = self.dt / (4 * self.dv) * (
                        (self.v - self.u_rest) / self.tau_mem[1] + v_in_ie[i] * (-self.c1ie + self.c2ie_v) + v_in_ii[
                    i] * (self.c1ii + self.c2ii_v))
            f2_inh = self.dt / (2 * self.dv ** 2) * (v_in_ie[i] * self.c2ie + v_in_ii[i] * self.c2ii)

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
            g_inh = rho_inh_delta[i] * (-v_in_ie[i] * Fie_v + v_in_ii[i] * Fii_v)

            # calculate firing rate
            r_inh[i] = v_in_ie[i] * (self.c2ie[-1] * rho_inh[-2, i] / self.dv + self.gamma_ie.sf(
                (self.u_thr - self.u_rest) / (self.u_exc - self.u_rest)) * rho_inh_delta[i])
            if r_inh[i] < 0:
                # print(f"WARNING: r_inh < 0 ! (r_inh = {r_inh[i]}) ... Setting r_inh to 0")
                r_inh[i] = 0
            r_inh_delayed[i + ref_inh_delta_idx] = r_inh[i]

            # update rho and rho_delta
            rho_inh[:, i + 1] = np.linalg.solve(A_inh, np.matmul(B_inh, rho_inh[:, i]))
            rho_inh[:, i + 1] += dt * g_inh
            rho_inh_delta[i + 1] = rho_inh_delta[i] + dt * (
                        -(v_in_ie[i] + v_in_ii[i]) * rho_inh_delta[i] + r_inh_delayed[i])

        self.v_in_ee = v_in_ee
        self.v_in_ei = v_in_ei
        self.v_in_ie = v_in_ie
        self.v_in_ii = v_in_ii
        self.r_exc_delayed = r_exc_delayed
        self.ref_exc_delta_idx = ref_exc_delta_idx


        rho_plot_exc = rho_exc
        rho_plot_exc[v_reset_idx, :] = rho_exc[v_reset_idx, :] + rho_exc_delta[:]
        r_plot_exc = r_exc[:]

        rho_plot_inh = rho_inh
        rho_plot_inh[v_reset_idx, :] = rho_inh[v_reset_idx, :] + rho_inh_delta[:]
        r_plot_inh = r_inh[:]

        self.rho_exc_delta = rho_exc_delta
        self.rho_exc = rho_exc
        self.rho_inh = rho_inh
        self.rho_inh_delta = rho_inh_delta
        self.r_exc = r_exc
        self.r_inh = r_inh

        with h5py.File(self.name + '.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=self.t)
            h5file.create_dataset('v', data=self.v)
            h5file.create_dataset('r_exc', data=r_plot_exc)
            h5file.create_dataset('r_inh', data=r_plot_inh)
            h5file.create_dataset('rho_plot_exc', data=rho_plot_exc)
            h5file.create_dataset('rho_plot_inh', data=rho_plot_inh)

        # loop version of code
        self.get_diffusion_coeffs_1()

        # first init all arrays
        v_reset_idx = np.where(np.isclose(self.v, self.u_reset))[0][0]  # index of reset potential in array
        self.v_reset_idx = v_reset_idx
        ref_delta_idxs = np.array([int(self.tau_ref[k] / self.dt) for k in range(self.n_populations)])
        max_ref_delta_indx = np.max(ref_delta_idxs)
        rho = np.zeros((self.n_populations, len(self.v), len(self.t)))
        rho_delta = np.zeros((self.n_populations, len(self.t)))
        r = np.zeros((self.n_populations, len(self.t)))  # output firing rate
        r_delayed = np.zeros((self.n_populations, (len(self.t) + max_ref_delta_indx)))
        v_in = np.zeros((self.n_populations, self.n_populations, len(self.t))) # ref_exc_delta_idx used to be here

        r_conv = np.zeros(self.n_populations)
        exc_idxs = [i for i, type in enumerate(self.population_type) if type == 'exc']
        inh_idxs = [i for i, type in enumerate(self.population_type) if type == 'inh']

        for i in range(len(self.population_type)):
            # initialize arrays
            rho[i, :, 0] = scipy.stats.norm.pdf(self.v, self.u_rest, 1)
            rho[i, 0, 0] = 0
            rho[i, -1, 0] = 0

        # Determine population dynamics (diffusion approximation)
        for i, t_ in enumerate(tqdm(self.t[:-1], f'simulating Nykamp model for {self.t[:-1].shape[0]} time steps')):

            # if i > 0:
            #     r_conv = self.mat_convolve(r[:(i + 1), :], self.alpha)[:, :, -len(self.alpha)] * self.dt
            # v_in[:, :, i] = self.connectivity_matrix * r_conv + self.in2D

            for j, type_j in enumerate(self.population_type):

                # as of now r_conv has only one dimension, same as r
                # each entry is convoluted by its representing kernel
                if i > 0:
                    r_conv[j] = np.convolve(r[j, :(i + 1)], self.alpha)[-len(self.alpha)] * self.dt
                v_in[:, j, i] = self.connectivity_matrix[:, j] * r_conv + self.input[:, j, i]

                # v_in is the incoming FIRING RATE!
                # it is not a voltage so consider renaming it nu_in


                # coefficients for finite difference matrices
                # c1, c2 are over all v steps and i is a time step
                # here the values need to be probably split between excitatory and inhibitory types
                if type_j == 'exc':
                    c1ee = self.c[j, 0, 0]
                    c1ei = self.c[j, 0, 1]
                    c2ee = self.c[j, 1, 0]
                    c2ei = self.c[j, 1, 1]
                    c1ee_v = self.c_v[j, 0, 0]
                    c1ei_v = self.c_v[j, 0, 1]
                    c2ee_v = self.c_v[j, 1, 0]
                    c2ei_v = self.c_v[j, 1, 1]
                    # TODO: this can be collapsed into drawing out the coeffs, since they can be take out of the sum
                    f0_exc = self.dt / 2 * (1 / self.tau_mem[0] + np.sum(- v_in[exc_idxs, j, i]) * c1ee_v
                                            + np.sum(v_in[inh_idxs, j, i]) * c1ei_v)
                    f1_exc = self.dt / (4 * self.dv) * ((self.v - self.u_rest) / self.tau_mem[0] +
                                                        np.sum(v_in[exc_idxs, j, i]) * (-c1ee + c2ee_v) +
                                                        np.sum(v_in[inh_idxs, j, i]) * (c1ei + c2ei_v))
                    f2_exc = self.dt / (2 * self.dv ** 2) * (np.sum(v_in[exc_idxs, j, i]) * c2ee +
                                                             np.sum(v_in[inh_idxs, j, i]) * c2ei)

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
                    g_exc = rho_delta[j, i] * (-np.sum(v_in[exc_idxs, j, i]) * self.dFdv[j, 0] +
                                               np.sum(v_in[inh_idxs, j, i]) * self.dFdv[j, 1])

                    # calculate firing rate
                    r[j, i] = np.sum(v_in[exc_idxs, j, i]) * (c2ee[-1] * rho[j, -2, i] / self.dv +
                                                               self.gamma_funcs[j].sf((self.u_thr - self.u_rest) / (
                                                                           self.u_exc - self.u_rest)) *
                                                               rho_delta[j, i])
                    if r[j, i] < 0:
                        r[j, i] = 0
                    r_delayed[j, i + ref_delta_idxs[j]] = r[j, i]

                    rho[j, :, i + 1] = np.linalg.solve(A_exc, np.matmul(B_exc, rho[j, :, i]))
                    rho[j, :, i + 1] += self.dt * g_exc
                    rho_delta[j, i + 1] = rho_delta[j, i] + self.dt * (
                            -(np.sum(v_in[exc_idxs, j, i]) + np.sum(v_in[inh_idxs, j, i])) *
                            rho_delta[j, i] + r_delayed[j, i])
                else:
                    # inhibitory population
                    # ================================================================================================================

                    c1ie = self.c[j, 0, 0]
                    c1ii = self.c[j, 0, 1]
                    c2ie = self.c[j, 1, 0]
                    c2ii = self.c[j, 1, 1]
                    c1ie_v = self.c_v[j, 0, 0]
                    c1ii_v = self.c_v[j, 0, 1]
                    c2ie_v = self.c_v[j, 1, 0]
                    c2ii_v = self.c_v[j, 1, 1]

                    # coefficients for finite difference matrices
                    f0_inh = self.dt / 2 * (1 / self.tau_mem[1] - np.sum(v_in[exc_idxs, j, i]) * c1ie_v +
                                            np.sum(v_in[inh_idxs, j, i]) * c1ii_v)
                    f1_inh = self.dt / (4 * self.dv) * (
                            (self.v - self.u_rest) / self.tau_mem[1] + np.sum(v_in[exc_idxs, j, i]) * (-c1ie + c2ie_v) +
                            np.sum(v_in[inh_idxs, j, i]) * (c1ii + c2ii_v))
                    f2_inh = self.dt / (2 * self.dv ** 2) * (np.sum(v_in[exc_idxs, j, i]) * c2ie +
                                                             np.sum(v_in[inh_idxs, j, i]) * c2ii)

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
                    g_inh = rho_delta[j, i] * (-np.sum(v_in[exc_idxs, j, i]) * self.dFdv[j, 0] +
                                               np.sum(v_in[inh_idxs, j, i]) * self.dFdv[j, 1])

                    # calculate firing rate
                    if i == 250:
                        a=1
                    r[j, i] = np.sum(v_in[exc_idxs, j, i]) * (c2ie[-1] * rho[j, -2, i] / self.dv +
                                                              self.gamma_funcs[j].sf((self.u_thr - self.u_rest) / (
                                                                          self.u_exc - self.u_rest)) *
                                                              rho_delta[j, i])
                    if r[j, i] < 0:
                        # print(f"WARNING: r_inh < 0 ! (r_inh = {r_inh[i]}) ... Setting r_inh to 0")
                        r[j, i] = 0
                    r_delayed[j, i + ref_delta_idxs[j]] = r[j, i]

                    # update rho and rho_delta
                    rho[j, :, i + 1] = np.linalg.solve(A_inh, np.matmul(B_inh, rho[j, :, i]))
                    rho[j, :, i + 1] += dt * g_inh
                    rho_delta[j, i + 1] = rho_delta[j, i] + dt * (
                            -(np.sum(v_in[exc_idxs, j, i]) + np.sum(v_in[inh_idxs, j, i])) *
                            rho_delta[j, i] + r_delayed[j, i])

        rho_plot_exc = rho[0]
        rho_plot_exc[v_reset_idx, :] = rho[0, v_reset_idx, :] + rho_delta[0, :]
        r_plot_exc = r[0, :]

        rho_plot_inh = rho[1]
        rho_plot_inh[v_reset_idx, :] = rho[1, v_reset_idx, :] + rho_delta[1, :]
        r_plot_inh = r[1, :]

        if np.allclose(self.rho_exc, rho[0]) and np.allclose(self.rho_inh, rho[1]):

            print('works!')
        with h5py.File(self.name + '_1.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=self.t)
            h5file.create_dataset('v', data=self.v)
            h5file.create_dataset('r_exc', data=r_plot_exc)
            h5file.create_dataset('r_inh', data=r_plot_inh)
            h5file.create_dataset('rho_plot_exc', data=rho_plot_exc)
            h5file.create_dataset('rho_plot_inh', data=rho_plot_inh)

    def mat_convolve(self, x, kernel):
        """
        Function that convolves an array of time series with the same kernel unsing np.convolve

        Parameter
        ---------
        x : np.array of float (2D)
            input array
        kernel : np.array of float (1D)
            convolution kernel

        Returns
        -------
        B : np.array of float (2D)
            convolved input array
        """

        n_x = x.shape[0]
        n_y = x.shape[1]

        # use extra convolution to determine shape of output array
        n_b = np.convolve(x[0, 0], kernel).shape[0]

        b = np.zeros((n_x, n_y, n_b))
        for i in range(n_x):
            for j in range(n_y):
                b[i, j] = np.convolve(x[i, j], kernel)
        return b

    def get_diffusion_coeffs(self):

        # TODO: make these matrix operations, get all diffusion coeffs at once?
        v = np.arange(self.u_inh, self.u_thr + self.dv, self.dv)
        n_v = v.shape[0]

        n_coeff = 4 * self.n_populations
        self.diffusion_coefficients = np.zeros([n_coeff, n_v])
        self.diffusion_coefficients_dv = np.zeros([n_coeff, n_v])
        self.dFdv = np.zeros([int(n_coeff / 2), n_v])

        self.scales = []
        for k in range(self.n_populations):
            if self.population_type[k] == 'exc':

                # init arrays for diffuson coeffs
                c1ee = np.zeros(len(v))
                c2ee = np.zeros(len(v))
                c1ei = np.zeros(len(v))
                c2ei = np.zeros(len(v))

                # synapse parameters
                mu_ee = self.mu_gamma[k, 0]
                mu_ei = self.mu_gamma[k, 1]

                coeff_var_ee = self.var_coeff_gamma[k, 0]
                coeff_var_ei = self.var_coeff_gamma[k, 1]
                var_ee = (coeff_var_ee * mu_ee) ** 2
                var_ei = (coeff_var_ei * mu_ei) ** 2

                scale_ee = var_ee / mu_ee
                scale_ei = var_ei / mu_ei

                self.scales.append(scale_ee)
                self.scales.append(scale_ei)

                # conductance jump distributions
                gamma_ee = gamma(a=coeff_var_ee ** (-2), loc=0, scale=scale_ee)
                gamma_ei = gamma(a=coeff_var_ei ** (-2), loc=0, scale=scale_ei)

                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
                    int_c1ee = gamma_ee.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ee = int_c1ee * (v_ - vpe)
                    c1ee[i] = np.trapz(x=vpe, y=int_c1ee)
                    c2ee[i] = np.trapz(x=vpe, y=int_c2ee)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                        int_c1ei = gamma_ei.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ei = int_c1ei * (vpi - v_)
                        c1ei[i] = np.trapz(x=vpi, y=int_c1ei)
                        c2ei[i] = np.trapz(x=vpi, y=int_c2ei)

                c1ei[0] = c1ei[1]
                c2ei[0] = 0

                self.diffusion_coefficients[4*k:4*(k+1)] = np.array([c1ee, c2ee, c1ei, c2ei])
                self.gamma_ee = gamma_ee
                Fee_v = np.gradient(gamma_ee.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                        * np.heaviside(self.v - self.u_rest, 0.5)
                Fei_v = np.gradient(gamma_ei.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                        * np.heaviside(self.u_rest - self.v, 0.5)

                self.dFdv[2*k:2*(k+1)] = np.array([Fee_v, Fei_v])


            elif self.population_type[k] == 'inh':


                # init arrays for diffusion coeffs
                c1ie = np.zeros(len(v))
                c2ie = np.zeros(len(v))
                c1ii = np.zeros(len(v))
                c2ii = np.zeros(len(v))

                # synapse parameters
                mu_ie = self.mu_gamma[k, 0]
                mu_ii = self.mu_gamma[k, 1]

                coeff_var = self.var_coeff_gamma[k, 0]
                var_ie = (coeff_var * mu_ie) ** 2
                var_ii = (coeff_var * mu_ii) ** 2

                scale_ie = var_ie / mu_ie
                scale_ii = var_ii / mu_ii

                # conductance jump distributions
                gamma_ie = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ie)
                gamma_ii = gamma(a=coeff_var ** (-2), loc=0, scale=scale_ii)

                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
                    int_c1ie = gamma_ie.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ie = int_c1ie * (v_ - vpe)
                    c1ie[i] = np.trapz(x=vpe, y=int_c1ie)
                    c2ie[i] = np.trapz(x=vpe, y=int_c2ie)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                        int_c1ii = gamma_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ii = int_c1ii * (vpi - v_)
                        c1ii[i] = np.trapz(x=vpi, y=int_c1ii)
                        c2ii[i] = np.trapz(x=vpi, y=int_c2ii)


                c1ii[0] = c1ii[1]
                c2ii[0] = 0

                self.diffusion_coefficients[4*k:4*(k+1)] = c1ie, c2ie, c1ii, c2ii
                self.gamma_ie = gamma_ie
                Fie_v = np.gradient(gamma_ie.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                                   * np.heaviside(self.v - self.u_rest, 0.5)
                Fii_v = np.gradient(gamma_ii.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                                   * np.heaviside(self.u_rest - self.v, 0.5)
                self.dFdv[2*k:2*(k+1)] = np.array([Fie_v, Fii_v])

            else:
                raise NotImplementedError('population types must be "exc" or "inh"!')

    def get_diffusion_coeffs_1(self):

        # TODO: make these matrix operations, get all diffusion coeffs at once?
        v = np.arange(self.u_inh, self.u_thr + self.dv, self.dv)
        n_v = v.shape[0]

        self.c = np.zeros((self.n_populations, 2, 2, n_v))
        self.c_v = np.zeros_like(self.c)
        self.dFdv = np.zeros([self.n_populations, 2, n_v])

        self.gamma_funcs = []
        for k in range(self.n_populations):
            if self.population_type[k] == 'exc':

                # init arrays for diffuson coeffs
                c1ee = np.zeros(len(v))
                c2ee = np.zeros(len(v))
                c1ei = np.zeros(len(v))
                c2ei = np.zeros(len(v))

                # conductance jump distributions
                # input synapse parameters into gamma distribution
                gamma_ee = gamma(a=self.var_coeff_gamma[k, 0] ** (-2), loc=0,
                                 scale=self.var_coeff_gamma[k, 0]**2*self.mu_gamma[k, 0])
                gamma_ei = gamma(a=self.var_coeff_gamma[k, 1] ** (-2),
                                 loc=0, scale=self.var_coeff_gamma[k, 1]**2*self.mu_gamma[k, 1])

                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
                    int_c1ee = gamma_ee.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ee = int_c1ee * (v_ - vpe)
                    c1ee[i] = np.trapz(x=vpe, y=int_c1ee)
                    c2ee[i] = np.trapz(x=vpe, y=int_c2ee)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                        int_c1ei = gamma_ei.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ei = int_c1ei * (vpi - v_)
                        c1ei[i] = np.trapz(x=vpi, y=int_c1ei)
                        c2ei[i] = np.trapz(x=vpi, y=int_c2ei)

                c1ei[0] = c1ei[1]
                c2ei[0] = 0

                self.c[k] = np.array([[[c1ee, c1ei], [c2ee, c2ei]]])
                c1ee_v = np.gradient(c1ee, self.v)
                c1ei_v = np.gradient(c1ei, self.v)
                c2ee_v = np.gradient(c2ee, self.v)
                c2ei_v = np.gradient(c2ei, self.v)
                self.c_v[k] = np.array([[[c1ee_v, c1ei_v], [c2ee_v, c2ei_v]]])
                self.gamma_funcs.append(gamma_ee)
                Fee_v = np.gradient(gamma_ee.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                        * np.heaviside(self.v - self.u_rest, 0.5)
                Fei_v = np.gradient(gamma_ei.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                        * np.heaviside(self.u_rest - self.v, 0.5)

                self.dFdv[k] = np.array([Fee_v, Fei_v])


            elif self.population_type[k] == 'inh':


                # init arrays for diffusion coeffs
                c1ie = np.zeros(len(v))
                c2ie = np.zeros(len(v))
                c1ii = np.zeros(len(v))
                c2ii = np.zeros(len(v))

                # conductance jump distributions
                gamma_ie = gamma(a=self.var_coeff_gamma[k, 0] ** (-2), loc=0,
                                 scale=self.var_coeff_gamma[k, 0] ** 2 * self.mu_gamma[k, 0])
                gamma_ii = gamma(a=self.var_coeff_gamma[k, 1] ** (-2),
                                 loc=0, scale=self.var_coeff_gamma[k, 1] ** 2 * self.mu_gamma[k, 1])

                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
                    int_c1ie = gamma_ie.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ie = int_c1ie * (v_ - vpe)
                    c1ie[i] = np.trapz(x=vpe, y=int_c1ie)
                    c2ie[i] = np.trapz(x=vpe, y=int_c2ie)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                        int_c1ii = gamma_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ii = int_c1ii * (vpi - v_)
                        c1ii[i] = np.trapz(x=vpi, y=int_c1ii)
                        c2ii[i] = np.trapz(x=vpi, y=int_c2ii)

                c1ii[0] = c1ii[1]
                c2ii[0] = 0

                self.c[k] = np.array([[c1ie, c1ii], [c2ie, c2ii]])
                c1ie_v = np.gradient(c1ie, self.v)
                c1ii_v = np.gradient(c1ii, self.v)
                c2ie_v = np.gradient(c2ie, self.v)
                c2ii_v = np.gradient(c2ii, self.v)
                self.c_v[k] = np.array([[[c1ie_v, c1ii_v], [c2ie_v, c2ii_v]]])
                self.gamma_funcs.append(gamma_ie)
                Fie_v = np.gradient(gamma_ie.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                                   * np.heaviside(self.v - self.u_rest, 0.5)
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

class Nykamp_Model_1():
    """
    Class for the population model of Nykamp and Tranchina
    Probability flux over time is used to model synaptic membrane and firing rates of neuron
    populations
    """

    def __init__(self, parameters, name='Nykamp'):

        self.connectivity_matrix = np.ones([2, 2])
        self.tau_alpha = 1/3
        self.n_alpha = 9

        self.dt = 0.1
        self.dv = 0.01
        self.T = 100

        if name is None:
            self.name = 'Nykamp_example'
        else:
            self.name = name
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
            self.u_exc = parameters['u_exc']
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

        self.n_populations = len(self.population_type)

        # input current
        if 'input_type' in parameters:
            self.input_type = parameters['input_type']
        else:
            self.input_type = 'rate'
        if 'g_leak' in parameters:
            self.g_leak = parameters['g_leak']
        else:
            self.g_leak = [1]*self.n_populations



    def simulate(self, dv=None, dt=None, T=None, dv_fine=None, sparse_mat=True, verbose=0):

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

        self.sparse_mat = sparse_mat

        self.t = np.arange(0, T, dt)
        self.v = np.arange(self.u_inh, self.u_thr + dv, dv)

        self.input = np.zeros([self.n_populations, self.n_populations, self.t.shape[0]])
        self.i_ext = np.zeros([self.n_populations, self.t.shape[0]])
        if self.input_type == 'rate':
            # set up input_function for rate over time
            self.input[self.input_function_idx[0], self.input_function_idx[1]] = self.input_function(t=self.t)
        elif self.input_type == 'current':
            self.i_ext[self.input_function_idx] = self.input_function(t=self.t)

        # calculate alpha kernel
        self.get_alpha_kernel()

        if verbose > 0:
            t0_coeff = time.time()

        # loop version of code
        self.get_diffusion_coeffs()
        self.r = None

        if verbose > 0:
            t1_coeff = time.time()
            print(f'time for coeffs: {t1_coeff - t0_coeff:.3f}s')

        # first init all arrays
        v_reset_idx = np.where(np.isclose(self.v, self.u_reset))[0][0]  # index of reset potential in array
        self.v_reset_idx = v_reset_idx
        ref_delta_idxs = np.array([int(self.tau_ref[k] / self.dt) for k in range(self.n_populations)])
        max_ref_delta_indx = np.max(ref_delta_idxs)
        rho = np.zeros((self.n_populations, len(self.v), len(self.t)))
        rho_delta = np.zeros((self.n_populations, len(self.t)))
        r = np.zeros((self.n_populations, len(self.t)))  # output firing rate
        r_delayed = np.zeros((self.n_populations, (len(self.t) + max_ref_delta_indx)))
        v_in = np.zeros((self.n_populations, self.n_populations, len(self.t))) # ref_exc_delta_idx used to be here

        r_conv = np.zeros(self.n_populations)
        exc_idxs = [i for i, type in enumerate(self.population_type) if type == 'exc']
        inh_idxs = [i for i, type in enumerate(self.population_type) if type == 'inh']

        self.c1eext = np.zeros_like(self.v)
        self.c2eext = np.zeros_like(self.v)
        self.c1eext_v = np.zeros_like(self.v)
        self.c2eext_v = np.zeros_like(self.v)

        for i in range(len(self.population_type)):
            # initialize arrays
            rho[i, :, 0] = scipy.stats.norm.pdf(self.v, self.u_rest, 1)
            rho[i, 0, 0] = 0
            rho[i, -1, 0] = 0

        # Determine population dynamics (diffusion approximation)
        for i, t_ in enumerate(tqdm(self.t[:-1],f"simulating {self.population_type} neuron populations for"
                                                  f" {self.t[:-1].shape[0]} time steps")):

            # if i > 0:
            #     r_conv = self.mat_convolve(r[:(i + 1), :], self.alpha)[:, :, -len(self.alpha)] * self.dt
            # v_in[:, :, i] = self.connectivity_matrix * r_conv + self.in2D

            # if self.voltage_idx is not None:
            #     # map shift to dv
            #     v_shift = int(self.input_voltage[i] / self.dv)
            #     rho[self.voltage_idx, :, i] = np.roll(rho[self.voltage_idx, :, i], v_shift)


            for j, type_j in enumerate(self.population_type):

                # as of now r_conv has only one dimension, same as r
                # each entry is convoluted by its representing kernel
                if i > 0:
                    r_conv[j] = np.convolve(r[j, :(i + 1)], self.alpha)[-len(self.alpha)] * self.dt
                v_in[:, j, i] = self.connectivity_matrix[:, j] * r_conv + self.input[:, j, i]

                # coefficients for finite difference matrices
                # c1, c2 are over all v steps and i is a time step
                # here the values are split between excitatory and inhibitory types
                if type_j == 'exc':
                    # excitatory population
                    # ================================================================================================================
                    c1ee = self.c[j, 0, 0]
                    c1ei = self.c[j, 0, 1]
                    c2ee = self.c[j, 1, 0]
                    c2ei = self.c[j, 1, 1]
                    c1ee_v = self.c_v[j, 0, 0]
                    c1ei_v = self.c_v[j, 0, 1]
                    c2ee_v = self.c_v[j, 1, 0]
                    c2ei_v = self.c_v[j, 1, 1]

                    if self.input_type == 'current':
                        v_ext = self.i_ext[j, i] / self.g_leak[j]
                        mask1 = np.where(self.v < v_ext + self.u_inh)[0]
                        mask2 = np.where(self.v > v_ext + self.u_inh)[0]

                        self.c1eext[mask1] = self.v[mask1] - self.u_inh
                        self.c1eext[mask2] = v_ext
                        self.c1eext_v[mask1] = 1
                        self.c1eext_v[mask2] = 0

                        self.c2eext[mask1] = 0.5 * (self.v[mask1] - self.u_inh) ** 2
                        self.c2eext[mask2] = v_ext ** 2
                        self.c2eext_v[mask1] = (self.v[mask1] - self.u_inh)
                        self.c2eext_v[mask2] = 0

                    # TODO: this can be collapsed into drawing out the coeffs, since they can be taken out of the sum
                    #  check if this is correct
                    f0_exc = self.dt / 2 * (1 / self.tau_mem[0] + np.sum(- v_in[exc_idxs, j, i]) * c1ee_v
                                            + np.sum(v_in[inh_idxs, j, i]) * c1ei_v - self.c1eext_v)
                    f1_exc = self.dt / (4 * self.dv) * ((self.v - self.u_rest) / self.tau_mem[0] +
                                                        - self.c1eext + self.c2eext_v +  # new external inputs
                                                        np.sum(v_in[exc_idxs, j, i]) * (-c1ee + c2ee_v) +
                                                        np.sum(v_in[inh_idxs, j, i]) * (c1ei + c2ei_v))
                    f2_exc = self.dt / (2 * self.dv ** 2) * (np.sum(v_in[exc_idxs, j, i]) * c2ee +
                                                             np.sum(v_in[inh_idxs, j, i]) * c2ei + self.c2eext)

                    if i == 0 and verbose > 0:
                        time0_A_exc = time.time()

                    A_exc = self.get_A(f0_exc, f1_exc, f2_exc)

                    if i == 0 and verbose > 0:
                        time0_B_exc = time.time()
                        time1_A_exc = time.time()

                    B_exc = self.get_B(f0_exc, f1_exc, f2_exc)

                    if i == 0 and verbose > 0:
                        time1_B_exc = time.time()
                        time0_rho_exc = time.time()

                    # contribution to drho/dt from rho_delta at u_res
                    g_exc = rho_delta[j, i] * (-np.sum(v_in[exc_idxs, j, i]) * self.dFdv[j, 0] +
                                               np.sum(v_in[inh_idxs, j, i]) * self.dFdv[j, 1])

                    # calculate firing rate
                    r_j = np.sum(v_in[exc_idxs, j, i]) * (c2ee[-1] * rho[j, -2, i] / self.dv +
                                                               self.gamma_funcs[j].sf((self.u_thr - self.u_rest) / (
                                                                           self.u_exc - self.u_rest)) *
                                                               rho_delta[j, i])
                    r_ext = + self.c2eext[-1] * rho[j, -2, i] / self.dv

                    r[j, i] = r_j + r_ext

                    if r[j, i] > 1e3:
                        a=1
                    if r[j, i] < 0:
                        r[j, i] = 0
                    if not r[j, i] < 0 and not r[j, i] >0 and not r[j, i] == 0:
                        a=1
                    r_delayed[j, i + ref_delta_idxs[j]] = r[j, i]

                    if not self.sparse_mat:
                        rho[j, :, i + 1] = np.linalg.solve(A_exc, np.matmul(B_exc, rho[j, :, i]))
                    else:
                        rho[j, :, i + 1] = scipy.sparse.linalg.spsolve(A_exc, B_exc.dot(rho[j, :, i]))

                    rho[j, :, i + 1] += self.dt * g_exc
                    rho_delta[j, i + 1] = rho_delta[j, i] + self.dt * (
                            -(np.sum(v_in[exc_idxs, j, i]) + np.sum(v_in[inh_idxs, j, i])) *
                            rho_delta[j, i] + r_delayed[j, i])

                    if i == 0 and verbose > 0:
                        time1_rho_exc = time.time()

                else:
                    # inhibitory population
                    # ================================================================================================================

                    c1ie = self.c[j, 0, 0]
                    c1ii = self.c[j, 0, 1]
                    c2ie = self.c[j, 1, 0]
                    c2ii = self.c[j, 1, 1]
                    c1ie_v = self.c_v[j, 0, 0]
                    c1ii_v = self.c_v[j, 0, 1]
                    c2ie_v = self.c_v[j, 1, 0]
                    c2ii_v = self.c_v[j, 1, 1]

                    # coefficients for finite difference matrices
                    f0_inh = self.dt / 2 * (1 / self.tau_mem[1] - np.sum(v_in[exc_idxs, j, i]) * c1ie_v +
                                            np.sum(v_in[inh_idxs, j, i]) * c1ii_v)
                    f1_inh = self.dt / (4 * self.dv) * (
                            (self.v - self.u_rest) / self.tau_mem[1] +
                            # - (self.i_ext[j, i] / self.g_leak[j]) +
                            np.sum(v_in[exc_idxs, j, i]) * (-c1ie + c2ie_v) +
                            np.sum(v_in[inh_idxs, j, i]) * (c1ii + c2ii_v))
                    f2_inh = self.dt / (2 * self.dv ** 2) * (np.sum(v_in[exc_idxs, j, i]) * c2ie +
                                                             np.sum(v_in[inh_idxs, j, i]) * c2ii)

                    if i == 0 and verbose > 0:
                        time0_A_inh = time.time()

                    A_inh = self.get_A(f0_inh, f1_inh, f2_inh)

                    if i == 0 and verbose > 0:
                        time0_B_inh = time.time()
                        time1_A_inh = time.time()

                    B_inh = self.get_B(f0_inh, f1_inh, f2_inh)

                    if i == 0 and verbose > 0:
                        time1_B_inh = time.time()
                        time0_rho_inh = time.time()

                    # contribution to drho/dt from rho_delta at u_res
                    g_inh = rho_delta[j, i] * (-np.sum(v_in[exc_idxs, j, i]) * self.dFdv[j, 0] +
                                               np.sum(v_in[inh_idxs, j, i]) * self.dFdv[j, 1])

                    # calculate firing rate
                    r_j = np.sum(v_in[exc_idxs, j, i]) * (c2ie[-1] * rho[j, -2, i] / self.dv +
                                                              self.gamma_funcs[j].sf((self.u_thr - self.u_rest) / (
                                                                          self.u_exc - self.u_rest)) *
                                                              rho_delta[j, i])
                    r_ext = 0 # c1ie[-1]*(1/self.g_leak[j])*self.i_ext[j, i] * rho[j, -2, i] / self.dv

                    r[j, i] = r_j + r_ext
                    if r[j, i] < 0:
                        # print(f"WARNING: r_inh < 0 ! (r_inh = {r_inh[i]}) ... Setting r_inh to 0")
                        r[j, i] = 0
                    r_delayed[j, i + ref_delta_idxs[j]] = r[j, i]

                    # update rho and rho_delta
                    if not self.sparse_mat:
                        rho[j, :, i + 1] = np.linalg.solve(A_inh, np.matmul(B_inh, rho[j, :, i]))
                    else:
                        rho[j, :, i + 1] = scipy.sparse.linalg.spsolve(A_inh, B_inh.dot(rho[j, :, i]))
                    rho[j, :, i + 1] += dt * g_inh
                    rho_delta[j, i + 1] = rho_delta[j, i] + dt * (
                            -(np.sum(v_in[exc_idxs, j, i]) + np.sum(v_in[inh_idxs, j, i])) *
                            rho_delta[j, i] + r_delayed[j, i])

                    if i == 0 and verbose > 0:
                        time1_rho_inh = time.time()

            if i == 1 and verbose > 0:
                print('\n')
                if 'exc' in self.population_type:
                    time_A_exc = time1_A_exc - time0_A_exc
                    time_B_exc = time1_B_exc - time0_B_exc
                    time_rho_exc = time1_rho_exc - time0_rho_exc

                    print(f'time for A_exc: {time_A_exc:.3f}s')
                    print(f'time for B_exc: {time_B_exc:.3f}s')
                    print(f'time for rho_exc: {time_rho_exc:.3f}s')
                if 'inh' in self.population_type:
                    time_A_inh = time1_A_inh - time0_A_inh
                    time_B_inh = time1_B_inh - time0_B_inh
                    time_rho_inh = time1_rho_inh - time0_rho_inh

                    print(f'time for A_inh: {time_A_inh:.3f}s')
                    print(f'time for B_inh: {time_B_inh:.3f}s')
                    print(f'time for rho_inh: {time_rho_inh:.3f}s')

        rho_plot = np.zeros_like(rho)
        self.r = r
        for k in range(self.n_populations):
            rho_plot[k, :] = rho[k, :]
            rho_plot[k, v_reset_idx, :] = rho[k, v_reset_idx, :] + rho_delta[k, :]


        with h5py.File(self.name + '.hdf5', 'w') as h5file:
            h5file.create_dataset('t', data=self.t)
            h5file.create_dataset('v', data=self.v)
            h5file.create_dataset('r', data=r)
            h5file.create_dataset('rho_plot', data=rho_plot)
            h5file.create_dataset('p_types', data=self.population_type)

    def get_A(self, f0, f1, f2):
        if not self.sparse_mat:
            A = np.diag(1 + 2 * f2 - f0) + np.diagflat((-f2 - f1)[:-1], 1) + np.diagflat(
                (f1 - f2)[1:], -1)
            A[0, 1] = -2 * f1[1]
            A[-1, -2] = 2 * f1[-2]
        else:
            n_v = f0.shape[0]
            main = 1 + 2 * f2 - f0
            lower = (f1 - f2)[1:]
            upper = (-f2 - f1)[:-1]
            # Insert boundary conditions
            lower[-1] = 2 * f1[-2]
            upper[0] = -2 * f1[1]

            A = scipy.sparse.diags(
                diagonals=[main, lower, upper],
                offsets=[0, -1, 1], shape=(n_v, n_v),
                format='csr')
        return A

    def get_B(self, f0, f1, f2):
        if not self.sparse_mat:

            B = np.diag(1 - 2 * f2 + f0) + np.diagflat((f2 + f1)[:-1], 1) + np.diagflat(
                (f2 - f1)[1:], -1)
            B[0, 1] = 2 * f1[1]
            B[-1, -2] = -2 * f1[-2]
        else:
            n_v = f0.shape[0]
            main = 1 - 2 * f2 + f0
            upper = (f2 + f1)[:-1]
            lower = (f2 - f1)[1:]
            # Insert boundary conditions
            upper[0] = 2 * f1[1]
            lower[-1] = -2 * f1[-2]

            B = scipy.sparse.diags(
                diagonals=[main, lower, upper],
                offsets=[0, -1, 1], shape=(n_v, n_v),
                format='csr')
            # B_ = B.todense()
        return B

    def mat_convolve(self, x, kernel):
        """
        Function that convolves an array of time series with the same kernel unsing np.convolve

        Parameter
        ---------
        x : np.array of float (2D)
            input array
        kernel : np.array of float (1D)
            convolution kernel

        Returns
        -------
        B : np.array of float (2D)
            convolved input array
        """

        n_x = x.shape[0]
        n_y = x.shape[1]

        # use extra convolution to determine shape of output array
        n_b = np.convolve(x[0, 0], kernel).shape[0]

        b = np.zeros((n_x, n_y, n_b))
        for i in range(n_x):
            for j in range(n_y):
                b[i, j] = np.convolve(x[i, j], kernel)
        return b

    def get_diffusion_coeffs(self):

        v = np.arange(self.u_inh, self.u_thr + self.dv, self.dv)
        n_v = v.shape[0]

        self.c = np.zeros((self.n_populations, 2, 2, n_v))
        self.c_v = np.zeros_like(self.c)
        self.dFdv = np.zeros([self.n_populations, 2, n_v])

        self.gamma_funcs = []
        for k in range(self.n_populations):
            if self.population_type[k] == 'exc':

                # init arrays for diffuson coeffs
                c1ee = np.zeros(len(v))
                c2ee = np.zeros(len(v))
                c1ei = np.zeros(len(v))
                c2ei = np.zeros(len(v))

                # conductance jump distributions
                # input synapse parameters into gamma distribution
                gamma_ee = gamma(a=self.var_coeff_gamma[k, 0] ** (-2), loc=0,
                                 scale=self.var_coeff_gamma[k, 0]**2*self.mu_gamma[k, 0])
                gamma_ei = gamma(a=self.var_coeff_gamma[k, 1] ** (-2),
                                 loc=0, scale=self.var_coeff_gamma[k, 1]**2*self.mu_gamma[k, 1])

                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
                    int_c1ee = gamma_ee.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ee = int_c1ee * (v_ - vpe)
                    c1ee[i] = np.trapz(x=vpe, y=int_c1ee)
                    c2ee[i] = np.trapz(x=vpe, y=int_c2ee)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                        int_c1ei = gamma_ei.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ei = int_c1ei * (vpi - v_)
                        c1ei[i] = np.trapz(x=vpi, y=int_c1ei)
                        c2ei[i] = np.trapz(x=vpi, y=int_c2ei)

                c1ei[0] = c1ei[1]
                c2ei[0] = 0

                self.c[k] = np.array([[[c1ee, c1ei], [c2ee, c2ei]]])
                c1ee_v = np.gradient(c1ee, self.v)
                c1ei_v = np.gradient(c1ei, self.v)
                c2ee_v = np.gradient(c2ee, self.v)
                c2ei_v = np.gradient(c2ei, self.v)
                self.c_v[k] = np.array([[[c1ee_v, c1ei_v], [c2ee_v, c2ei_v]]])
                self.gamma_funcs.append(gamma_ee)
                Fee_v = np.gradient(gamma_ee.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                        * np.heaviside(self.v - self.u_rest, 0.5)
                Fei_v = np.gradient(gamma_ei.sf(x=(self.v - self.u_rest) / (self.u_inh - self.u_rest)), self.dv) \
                        * np.heaviside(self.u_rest - self.v, 0.5)

                self.dFdv[k] = np.array([Fee_v, Fei_v])


            elif self.population_type[k] == 'inh':


                # init arrays for diffusion coeffs
                c1ie = np.zeros(len(v))
                c2ie = np.zeros(len(v))
                c1ii = np.zeros(len(v))
                c2ii = np.zeros(len(v))

                # conductance jump distributions
                gamma_ie = gamma(a=self.var_coeff_gamma[k, 0] ** (-2), loc=0,
                                 scale=self.var_coeff_gamma[k, 0] ** 2 * self.mu_gamma[k, 0])
                gamma_ii = gamma(a=self.var_coeff_gamma[k, 1] ** (-2),
                                 loc=0, scale=self.var_coeff_gamma[k, 1] ** 2 * self.mu_gamma[k, 1])

                for i, v_ in enumerate(v):

                    vpe = np.arange(self.u_inh, v_ + self.dv, self.dv)
                    int_c1ie = gamma_ie.sf((v_ - vpe) / (self.u_exc - vpe))
                    int_c2ie = int_c1ie * (v_ - vpe)
                    c1ie[i] = np.trapz(x=vpe, y=int_c1ie)
                    c2ie[i] = np.trapz(x=vpe, y=int_c2ie)

                    if i > 0:
                        vpi = np.arange(v_, self.u_thr + self.dv, self.dv)
                        int_c1ii = gamma_ii.sf((v_ - vpi) / (self.u_inh - vpi))
                        int_c2ii = int_c1ii * (vpi - v_)
                        c1ii[i] = np.trapz(x=vpi, y=int_c1ii)
                        c2ii[i] = np.trapz(x=vpi, y=int_c2ii)

                c1ii[0] = c1ii[1]
                c2ii[0] = 0

                self.c[k] = np.array([[c1ie, c1ii], [c2ie, c2ii]])
                c1ie_v = np.gradient(c1ie, self.v)
                c1ii_v = np.gradient(c1ii, self.v)
                c2ie_v = np.gradient(c2ie, self.v)
                c2ii_v = np.gradient(c2ii, self.v)
                self.c_v[k] = np.array([[[c1ie_v, c1ii_v], [c2ie_v, c2ii_v]]])
                self.gamma_funcs.append(gamma_ie)
                Fie_v = np.gradient(gamma_ie.sf(x=(self.v - self.u_rest) / (self.u_exc - self.u_rest)), self.dv) \
                                   * np.heaviside(self.v - self.u_rest, 0.5)
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

    def plot(self, fname=None, heat_map=False, plot_idxs=None):

        if fname == None:
            fname = self.name

        with h5py.File(fname + '.hdf5', 'r') as h5file:

            t_plot = np.array(h5file['t'])
            v = np.array(h5file['v'])
            r_plot = np.array(h5file['r'])
            rho_plot = np.array(h5file['rho_plot'])
            p_types_raw = h5file['p_types']
            p_types = p_types_raw.asstr()[:]

        if plot_idxs is None:
            n_plots = len(p_types)
            plot_idxs = np.arange(n_plots)
        else:
            n_plots = len(plot_idxs)

        fig = plt.figure(figsize=(10, 4.25*n_plots))
        for i_plot, plot_idx in enumerate(plot_idxs):
            plot_loc_1 = int(2*i_plot + 1)
            plot_loc_2 = int(2 * i_plot + 2)
            if heat_map:
                ax = fig.add_subplot(n_plots, 2, plot_loc_1)
                X, Y = np.meshgrid(t_plot, v)
                z_min, z_max = 0, np.abs(rho_plot[plot_idx]).max()
                c = ax.pcolormesh(X, Y, rho_plot[plot_idx], cmap='viridis', vmin=z_min, vmax=z_max)
                fig.colorbar(c, ax=ax)

            else:
                ax = fig.add_subplot(n_plots, 2, plot_loc_1, projection='3d')
                X, Y = np.meshgrid(t_plot, v)
                ax.plot_surface(X, Y, rho_plot[plot_idx],
                                cmap="jet", linewidth=0, antialiased=False, rcount=100, ccount=100)
                ax.set_zlim3d(0, 1)

            ax.set_title(f"Membrane potential distribution ({str(p_types[plot_idx])})")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("membrane potential (mv)")

            ax = fig.add_subplot(n_plots, 2, plot_loc_2)
            ax.plot(t_plot, r_plot[plot_idx] * 1000)
            ax.set_title(f"Population activity ({str(p_types[plot_idx])})")
            ax.set_ylabel("Firing rate (Hz)")
            ax.set_xlabel("time (ms)")
            ax.grid()
        plt.tight_layout()
        plt.show()

    def clean(self):
        os.remove(self.name + '.hdf5')