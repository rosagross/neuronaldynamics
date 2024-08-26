"""
Pythonic implementation of Nykamp et al. 2000 population density model for 3 populations.
Author: Konstantin Weise
Edits and development: Aaron Miller
"""
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

heat_map = False

# @numba.njit(parallel=True)
def simulate():

    def alpha_func(t, alpha_bar=0.9995855855965002, tau_a=1/3, n_a=9):
        """
        Distribution of synaptic latencies (alpha kernel)
        Gamma function with mean 1/tau_a and order n_a

        Parameter
        ---------
        t : float or np.array of float [n_t]
            Time in ms (between 0 and 7.5 ms)
        alpha_bar : float
            Constant such that results integrates to 1
        tau_a : float
            Mean time constant corresponds to 1/tau_a
        n_a : int
            Order of gamma distribution

        Returns
        -------
        a : float or np.array of float [n_t]
            Distribution of synaptic latencies
        """

        a = alpha_bar * np.exp(-t / tau_a) / (tau_a * scipy.special.factorial(n_a - 1)) * (t / tau_a) ** (n_a - 1)

        if type(t) is np.ndarray:
            a[np.logical_not(np.logical_and(t > 0, t <= 7.5))] = 0
        else:
            if not (0 < t < 7.5):
                a = 0
        return a

    # parameters of synaptic latency distribution
    # tau_a = 1/3
    # n_a = 9
    # alpha_bar = 1/scipy.integrate.quad(alpha, 0, 7.5, args=(1, tau_a, n_a))[0]

    # simulation parameters
    T = 50 # 200
    dt = 0.1 # 0.1
    t = np.arange(0, T, dt)

    # input
    v0 = .7
    f = 10
    v_e_o = v0 * (1 + np.sin(2*np.pi*f*t/1000))

    # plt.plot(t, v_in_exc)
    # plt.plot(t, v_in_inh)

    # population parameters
    tau_exc_membrane = 20
    tau_inh_membrane = 10
    tau_exc_ref = 3
    tau_inh_ref = 1
    u_inh = -70
    u_res = -65
    u_thr = -55
    u_exc = 0
    u_reset = u_res
    verbose = True

    # synaptic connection weights
    W = 30
    w_ee = W/2      # exc -> exc
    w_ei = W        # exc -> inh
    w_ie = W        # inh -> exc
    w_ii = W        # inh -> inh

    # synapse parameters
    mu_exc_e = 0.008
    mu_exc_i = 0.027
    mu_inh_e = 0.020
    mu_inh_i = 0.066

    coeff_var = 0.5
    var_exc_e = (coeff_var * mu_exc_e)**2
    var_exc_i = (coeff_var * mu_exc_i)**2
    var_inh_e = (coeff_var * mu_inh_e)**2
    var_inh_i = (coeff_var * mu_inh_i)**2

    a_exc_e = mu_exc_e**2/var_exc_e
    a_exc_i = mu_exc_i**2/var_exc_i
    a_inh_e = mu_inh_e**2/var_inh_e
    a_inh_i = mu_inh_i**2/var_inh_i

    scale_exc_e = var_exc_e/mu_exc_e
    scale_exc_i = var_exc_i/mu_exc_i
    scale_inh_e = var_inh_e/mu_inh_e
    scale_inh_i = var_inh_i/mu_inh_i

    # conductance jump distributions
    gamma_exc_e = scipy.stats.gamma(a=a_exc_e, loc=0, scale=scale_exc_e)
    # TODO: replace
    # gamma_exc_e = gamma(a=coeff_var**(-2), loc=0, scale=scale_exc_e)
    gamma_exc_i = scipy.stats.gamma(a=a_exc_i, loc=0, scale=scale_exc_i)
    gamma_inh_e = scipy.stats.gamma(a=a_inh_e, loc=0, scale=scale_inh_e)
    gamma_inh_i = scipy.stats.gamma(a=a_inh_i, loc=0, scale=scale_inh_i)

    dv_fine = 0.01
    v_fine = np.arange(u_inh, u_thr+dv_fine, dv_fine)

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
        vpe = np.arange(u_inh, v_+dv_fine, dv_fine)
        int_exc_c1e = gamma_exc_e.sf((v_-vpe)/(u_exc-vpe))
        int_exc_c2e = int_exc_c1e * (v_-vpe)
        int_inh_c1e = gamma_inh_e.sf((v_-vpe)/(u_exc-vpe))
        int_inh_c2e = int_inh_c1e * (v_-vpe)

        c1e_exc_fine[i] = np.trapz(x=vpe, y=int_exc_c1e)
        c2e_exc_fine[i] = np.trapz(x=vpe, y=int_exc_c2e)
        c1e_inh_fine[i] = np.trapz(x=vpe, y=int_inh_c1e)
        c2e_inh_fine[i] = np.trapz(x=vpe, y=int_inh_c2e)

        if i > 0:
            vpi = np.arange(v_, u_thr+dv_fine, dv_fine)
            int_exc_c1i = gamma_exc_i.sf((v_-vpi)/(u_inh-vpi))
            int_exc_c2i = int_exc_c1i * (vpi-v_)
            int_inh_c1i = gamma_inh_i.sf((v_-vpi)/(u_inh-vpi))
            int_inh_c2i = int_inh_c1i * (vpi-v_)

            c1i_exc_fine[i] = np.trapz(x=vpi, y=int_exc_c1i)
            c2i_exc_fine[i] = np.trapz(x=vpi, y=int_exc_c2i)
            c1i_inh_fine[i] = np.trapz(x=vpi, y=int_inh_c1i)
            c2i_inh_fine[i] = np.trapz(x=vpi, y=int_inh_c2i)

    c1i_exc_fine[0] = c1i_exc_fine[1]
    c2i_exc_fine[0] = 0
    c1i_inh_fine[0] = c1i_inh_fine[1]
    c2i_inh_fine[0] = 0
    dc1e_exc_dv_fine = np.gradient(c1e_exc_fine, dv_fine)
    dc2e_exc_dv_fine = np.gradient(c2e_exc_fine, dv_fine)
    dc1i_exc_dv_fine = np.gradient(c1i_exc_fine, dv_fine)
    dc2i_exc_dv_fine = np.gradient(c2i_exc_fine, dv_fine)

    dc1e_inh_dv_fine = np.gradient(c1e_inh_fine, dv_fine)
    dc2e_inh_dv_fine = np.gradient(c2e_inh_fine, dv_fine)
    dc1i_inh_dv_fine = np.gradient(c1i_inh_fine, dv_fine)
    dc2i_inh_dv_fine = np.gradient(c2i_inh_fine, dv_fine)

    dv = 0.01
    v = np.arange(u_inh, u_thr+dv, dv)

    c1e_exc = np.interp(x=v, xp=v_fine, fp=c1e_exc_fine)
    c2e_exc = np.interp(x=v, xp=v_fine, fp=c2e_exc_fine)
    c1i_exc = np.interp(x=v, xp=v_fine, fp=c1i_exc_fine)
    c2i_exc = np.interp(x=v, xp=v_fine, fp=c2i_exc_fine)
    c1e_inh = np.interp(x=v, xp=v_fine, fp=c1e_inh_fine)
    c2e_inh = np.interp(x=v, xp=v_fine, fp=c2e_inh_fine)
    c1i_inh = np.interp(x=v, xp=v_fine, fp=c1i_inh_fine)
    c2i_inh = np.interp(x=v, xp=v_fine, fp=c2i_inh_fine)

    dc1e_exc_dv = np.interp(x=v, xp=v_fine, fp=dc1e_exc_dv_fine)
    dc2e_exc_dv = np.interp(x=v, xp=v_fine, fp=dc2e_exc_dv_fine)
    dc1i_exc_dv = np.interp(x=v, xp=v_fine, fp=dc1i_exc_dv_fine)
    dc2i_exc_dv = np.interp(x=v, xp=v_fine, fp=dc2i_exc_dv_fine)
    dc1e_inh_dv = np.interp(x=v, xp=v_fine, fp=dc1e_inh_dv_fine)
    dc2e_inh_dv = np.interp(x=v, xp=v_fine, fp=dc2e_inh_dv_fine)
    dc1i_inh_dv = np.interp(x=v, xp=v_fine, fp=dc1i_inh_dv_fine)
    dc2i_inh_dv = np.interp(x=v, xp=v_fine, fp=dc2i_inh_dv_fine)

    # alpha (synaptic latencies)
    t_alpha = t[t < 10]
    tau_alpha = 1/3
    n_alpha = 9
    alpha = np.exp(-t_alpha/tau_alpha) / (tau_alpha * scipy.special.factorial(n_alpha-1)) * (t_alpha/tau_alpha)**(n_alpha-1)
    alpha = alpha/np.trapz(alpha, dx=dt) # not necessary

    # contributions from delta distributions to rho_smooth
    dFe_exc_delta_dv = np.gradient(gamma_exc_e.sf(x=(v-u_res)/(u_exc-u_res)), dv) * np.heaviside(v-u_res, 0.5)
    dFi_exc_delta_dv = np.gradient(gamma_exc_i.sf(x=(v-u_res)/(u_inh-u_res)), dv) * np.heaviside(u_res-v, 0.5)
    dFe_inh_delta_dv = np.gradient(gamma_inh_e.sf(x=(v-u_res)/(u_exc-u_res)), dv) * np.heaviside(v-u_res, 0.5)
    dFi_inh_delta_dv = np.gradient(gamma_inh_i.sf(x=(v-u_res)/(u_inh-u_res)), dv) * np.heaviside(u_res-v, 0.5)

    # plt.plot(v, gamma_exc_e.sf(x=(v-u_res)/(u_exc-u_res)) * np.heaviside(v-u_res, 0.5), v, dFe_delta_dv)
    # plt.plot(v, gamma_exc_i.sf(x=(u_res-v)/(u_exc-u_res)) * np.heaviside(u_res-v, 0.5), v, dFi_delta_dv)

    # initialize arrays
    rho_exc = np.zeros((len(v), len(t)))                        # probability density of membrane potential
    rho_inh = np.zeros((len(v), len(t)))                        # probability density of membrane potential
    rho_exc_delta = np.zeros(len(t))                        # probability density of discontinuous membrane potential
    rho_inh_delta = np.zeros(len(t))                        # probability density of discontinuous membrane potential
    ref_exc_delta_idx = int(tau_exc_ref/dt)                     # number of time steps of refractory delay
    ref_inh_delta_idx = int(tau_inh_ref/dt)                     # number of time steps of refractory delay
    v_reset_idx = np.where(np.isclose(v, u_reset))[0][0]        # index of reset potential in array
    r_exc = np.zeros(len(t))                                    # output firing rate
    r_exc_delayed = np.zeros(len(t)+ref_exc_delta_idx)          # delayed output firing rate
    r_inh = np.zeros(len(t))                                    # output firing rate
    r_inh_delayed = np.zeros(len(t)+ref_inh_delta_idx)          # delayed output firing rate
    v_in_exc_exc = np.zeros(len(t))
    v_in_exc_inh = np.zeros(len(t))
    v_in_inh_exc = np.zeros(len(t))
    v_in_inh_inh = np.zeros(len(t))

    # initialize rho with a gaussian distribution around the resting potential
    rho_exc[:, 0] = scipy.stats.norm.pdf(v, u_res, 1)
    rho_exc[0, 0] = 0
    rho_exc[-1, 0] = 0
    rho_inh[:, 0] = rho_exc[:, 0]

    # Determine population dynamics (diffusion approximation)
    for i, t_ in enumerate(tqdm(t[:-1])):
    # for i, t_ in enumerate(t[:-1]):

        # excitatory population
        # ================================================================================================================
        # input to excitatory population
        if i > 0:
            r_exc_conv = np.convolve(r_exc[:(i+1)], alpha)[-len(alpha)]*dt
            r_inh_conv = np.convolve(r_inh[:(i+1)], alpha)[-len(alpha)]*dt
        else:
            r_exc_conv = 0
            r_inh_conv = 0

        v_in_exc_exc[i] = v_e_o[i] + w_ee * r_exc_conv
        v_in_exc_inh[i] = w_ie * r_inh_conv

        # coefficients for finite difference matrices
        # c1, c2 are over all v steps and i is a time step
        f0_exc = dt / 2 * (1/tau_exc_membrane - v_in_exc_exc[i]*dc1e_exc_dv + v_in_exc_inh[i]*dc1i_exc_dv)
        f1_exc = dt / (4*dv) * ((v-u_res)/tau_exc_membrane + v_in_exc_exc[i]*(-c1e_exc + dc2e_exc_dv) + v_in_exc_inh[i]*(c1i_exc + dc2i_exc_dv))
        f2_exc = dt / (2*dv**2) * (v_in_exc_exc[i]*c2e_exc + v_in_exc_inh[i]*c2i_exc)

        # LHS matrix (t+dt)
        A_exc = np.diag(1+2*f2_exc-f0_exc) + np.diagflat((-f2_exc-f1_exc)[:-1], 1) + np.diagflat((f1_exc-f2_exc)[1:], -1)
        A_exc[0, 1] = -2*f1_exc[1]
        A_exc[-1, -2] = 2*f1_exc[-2]

        # RHS matrix (t)
        B_exc = np.diag(1-2*f2_exc+f0_exc) + np.diagflat((f2_exc+f1_exc)[:-1], 1) + np.diagflat((f2_exc-f1_exc)[1:], -1)
        B_exc[0, 1] = 2*f1_exc[1]
        B_exc[-1, -2] = -2*f1_exc[-2]

        # contribution to drho/dt from rho_delta at u_res
        g_exc = rho_exc_delta[i] * (-v_in_exc_exc[i] * dFe_exc_delta_dv + v_in_exc_inh[i] * dFi_exc_delta_dv)

        # calculate firing rate
        r_exc[i] = v_in_exc_exc[i] * (c2e_exc[-1] * rho_exc[-2, i]/dv + gamma_exc_e.sf((u_thr-u_res)/(u_exc-u_res))*rho_exc_delta[i])
        if r_exc[i] < 0:
            # print(f"WARNING: r_exc < 0 ! (r_exc = {r_exc[i]}) ... Setting r_exc to 0")
            r_exc[i] = 0
        r_exc_delayed[i+ref_exc_delta_idx] = r_exc[i]

        # update rho and rho_delta
        # rho_exc[:, i+1] = np.linalg.solve(A_exc, np.matmul(B_exc, rho_exc[:, i][:, np.newaxis]))[:, 0]
        # old overly complicated version
        rho_exc[:, i + 1] = np.linalg.solve(A_exc, np.matmul(B_exc, rho_exc[:, i]))
        rho_exc[:, i+1] += dt * g_exc
        rho_exc_delta[i+1] = rho_exc_delta[i] + dt * (-(v_in_exc_exc[i] + v_in_exc_inh[i])*rho_exc_delta[i] + r_exc_delayed[i])

        # inhibitory population
        # ================================================================================================================
        # input to inhibitory population
        v_in_inh_exc[i] = w_ei * r_exc_conv
        v_in_inh_inh[i] = w_ii * r_inh_conv

        # coefficients for finite difference matrices
        f0_inh = dt / 2 * (1/tau_inh_membrane - v_in_inh_exc[i]*dc1e_inh_dv + v_in_inh_inh[i]*dc1i_inh_dv)
        f1_inh = dt / (4*dv) * ((v-u_res)/tau_inh_membrane + v_in_inh_exc[i]*(-c1e_inh + dc2e_inh_dv) + v_in_inh_inh[i]*(c1i_inh + dc2i_inh_dv))
        f2_inh = dt / (2*dv**2) * (v_in_inh_exc[i]*c2e_inh + v_in_inh_inh[i]*c2i_inh)

        # LHS matrix (t+dt)
        A_inh = np.diag(1+2*f2_inh-f0_inh) + np.diagflat((-f2_inh-f1_inh)[:-1], 1) + np.diagflat((f1_inh-f2_inh)[1:], -1)
        A_inh[0, 1] = -2*f1_inh[1]
        A_inh[-1, -2] = 2*f1_inh[-2]

        # RHS matrix (t)
        B_inh = np.diag(1-2*f2_inh+f0_inh) + np.diagflat((f2_inh+f1_inh)[:-1], 1) + np.diagflat((f2_inh-f1_inh)[1:], -1)
        B_inh[0, 1] = 2*f1_inh[1]
        B_inh[-1, -2] = -2*f1_inh[-2]

        # contribution to drho/dt from rho_delta at u_res
        g_inh = rho_inh_delta[i] * (-v_in_inh_exc[i] * dFe_inh_delta_dv + v_in_inh_inh[i] * dFi_inh_delta_dv)

        # calculate firing rate
        r_inh[i] = v_in_inh_exc[i] * (c2e_inh[-1] * rho_inh[-2, i]/dv + gamma_inh_e.sf((u_thr-u_res)/(u_exc-u_res))*rho_inh_delta[i])
        if r_inh[i] < 0:
            # print(f"WARNING: r_inh < 0 ! (r_inh = {r_inh[i]}) ... Setting r_inh to 0")
            r_inh[i] = 0
        r_inh_delayed[i+ref_inh_delta_idx] = r_inh[i]

        # update rho and rho_delta
        rho_inh[:, i + 1] = np.linalg.solve(A_inh, np.matmul(B_inh, rho_inh[:, i]))
        rho_inh[:, i+1] += dt * g_inh
        rho_inh_delta[i+1] = rho_inh_delta[i] + dt * (-(v_in_inh_exc[i] + v_in_inh_inh[i])*rho_inh_delta[i] + r_inh_delayed[i])

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


simulate()

with h5py.File('test' + '.hdf5', 'r') as h5file:

    t_plot = np.array(h5file['t'])
    v = np.array(h5file['v'])
    r_plot_exc = np.array(h5file['r_exc'])
    r_plot_inh = np.array(h5file['r_inh'])
    rho_plot_exc = np.array(h5file['rho_plot_exc'])
    rho_plot_inh = np.array(h5file['rho_plot_inh'])

#
# plt.plot(rho_plot_inh[:, 0])
# plt.plot(rho_plot_inh[:, 600])
# plt.plot(rho_plot_inh[:, 650])
# plt.plot(rho_plot_inh[:, 700])
# plt.plot(rho_plot_inh[:, 1000])
# plt.plot(rho_plot_inh[:, 1100])
# plt.plot(rho_plot_inh[:, 1200])
# plt.plot(rho_plot_inh[:, 1300])
# plt.plot(rho_plot_inh[:, 1400])
# plt.plot(rho_plot_inh[:, 1500])
# plt.plot(rho_plot_inh[:, 1600])
#
# plt.plot(r_plot_inh)



fig = plt.figure(figsize=(10, 8.5))

if heat_map:
    ax = fig.add_subplot(2, 2, 1)
    X, Y = np.meshgrid(t_plot, v)
    z_min, z_max = 0, np.abs(rho_plot_exc).max()
    c = ax.pcolormesh(X, Y, rho_plot_exc, cmap='viridis', vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax)

else:
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    X, Y = np.meshgrid(t_plot, v)
    ax.plot_surface(X, Y, rho_plot_exc,
                    cmap="jet", linewidth=0, antialiased=False, rcount=100, ccount=100)
    ax.set_zlim3d(0, 1)

ax.set_title("Membrane potential distribution (exc)")
ax.set_xlabel("time (ms)")
ax.set_ylabel("membrane potential (mv)")

ax = fig.add_subplot(2, 2, 2)
ax.plot(t_plot, r_plot_exc*1000)
ax.set_title("Population activity (exc)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_xlabel("time (ms)")
ax.grid()
plt.tight_layout()

if heat_map:
    ax = fig.add_subplot(2, 2, 3)
    X, Y = np.meshgrid(t_plot, v)
    z_min, z_max = 0, np.abs(rho_plot_inh).max()
    c = ax.pcolormesh(X, Y, rho_plot_inh, cmap='viridis', vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax)

else:
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    X, Y = np.meshgrid(t_plot, v)
    ax.plot_surface(X, Y, rho_plot_inh,
                    cmap="jet", linewidth=0, antialiased=False, rcount=100, ccount=100)
    ax.set_zlim3d(0, 1)

ax.set_title("Membrane potential distribution (inh)")
ax.set_xlabel("time (ms)")
ax.set_ylabel("membrane potential (mv)")


ax = fig.add_subplot(2, 2, 4)
ax.plot(t_plot, r_plot_inh*1000)
ax.set_title("Population activity (inh)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_xlabel("time (ms)")
ax.grid()
plt.tight_layout()
plt.show()

# plt.savefig(f"/home/kporzig/Desktop/Nykamp_network_A_dv_{dv}_dt_{dt}.jpg", dpi=600)

