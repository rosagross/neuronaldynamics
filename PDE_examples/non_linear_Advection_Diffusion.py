import numpy as np
from scipy.stats import norm
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('TkAgg')

solver = 'Hu-2021' # 'LW', 'Hu-2021'

def M(x, a, alpha, x_L=0.2):
    """
    compute Scharfetter-Gummel flux functional
    :param x: x
    :param a: drift
    :param alpha: diffusion
    :param x_L: resting point
    :return: M: Scharfetter-Gummel flux functional
    """
    # U = ((1/2)*x - x_L - a)*(x/alpha)
    # U = ((1 / 2) * x) * (x / alpha)
    # U = (x**2/2 - x_L*x)/alpha
    U = - a*x/alpha
    return np.exp(-U)

def harm_mean(x1, x2):
    return 2/((1/x1)+(1/x2))

def SG(x, a, x_L, dx, alpha):
    h = -(x-x_L) + a
    sg = 1/(np.exp(-h*dx/alpha)-1)
    return sg

dx_ = 0.002
dt = 0.02
T = 20
x0 = -66
sigma0 = 0.01
tau = 20
# alpha = 0.3
alpha = 0.1/tau
a = 8e4/tau
# x_ = np.arange(-70, -55, dx_)
x_ = np.arange(-70, -55, dx_)
t = np.arange(0, T, dt)
x_rest_ = -65
x_reset_ = -64
x_thr_idx = -1
# map x to 0,1
x = np.linspace(0, 1, x_.shape[0])
dx = np.diff(x)[0]
x_rest_idx_orig = np.where(x_ > x_rest_)[0][0] - 1
x_rest = x[x_rest_idx_orig]

x_reset_idx_orig = np.where(x_ > x_reset_)[0][0] - 1
x_reset = x[x_reset_idx_orig]
reset_idx = np.where(x == x_reset)

x_range_orig = x_.max() - x_.min()
x_range_new = 1.0

sigma0_new = sigma0 * (x_range_new/x_range_orig)
x0_idx_original = np.where(x_ > x0)[0][0] - 1
x0_new = x[x0_idx_original]

f_init = norm.pdf(x, x0_new, sigma0_new)
f_init[0] = f_init[-1] = 0
f_init /= (f_init.sum())
Nx = x.shape[0]
Nt = t.shape[0]

# a = np.zeros(Nt)
# a = 5 * (2*np.ones(Nt) + 1.5*np.sin(t/50)) + 5 * np.exp(-(t - 1.2)**2/0.1)
# a = a * ((2*np.ones(Nt) + 1.5*np.sin(t/3)) + np.exp(-(t - 1.2)**2/0.1))
c_out = alpha * dt / dx
alpha = np.ones(Nt)*alpha
t_off = 10


########################################################################################################################
# Test version on (0, 1)
########################################################################################################################

# dx=0.005
# x = np.arange(0, 1, dx)
# x_ = x.copy()
# x_reset = 0.2
# x_rest = 0.2
# f_init = norm.pdf(x, 0.3, 0.2)
# f_init[0] = f_init[-1] = 0
# f_init /= (f_init.sum())
# T=30
# t = np.arange(0, T, dt)
# Nx = x.shape[0]
# Nt = t.shape[0]
# alpha = np.ones(Nt)*0.3
# a = 50 * (2*np.ones(Nt) + 1.5*np.sin(t/3)) + 50 * np.exp(-(t - 1.2)**2/0.1)
#
# c_out = alpha[0] * dt / dx



if T > t_off:
    t_off_idx = int(np.where(t>t_off)[0][0] - 1)
    if not isinstance(a, np.ndarray):
        a = np.ones(Nt)*a
    a[t_off_idx:] = 0
# alpha[-100:] = 0

start = time.time()
if solver == 'LW':
    if not isinstance(a, np.ndarray) and not isinstance(alpha, np.ndarray):

        # Representation of sparse matrix and right-hand side
        main = np.zeros(Nx + 1)
        lower = np.zeros(Nx)
        upper = np.zeros(Nx)
        b = np.zeros(Nx + 1)

        # discretization for Lax-Wendroff (forward?)
        c = a*dt/dx
        s = alpha*dt/dx**2
        r1 = 0.5*(2*s + c + c**2)
        r2 = 1 - 2*s - c**2
        r3 = 0.5*(2*s - c + c**2)

        # Precompute sparse matrix
        main[:] = r2
        lower[:] = r1
        upper[:] = r3
        # Insert boundary conditions
        main[0] = 1
        main[Nx] = 1
        A = scipy.sparse.diags(diagonals=[main, lower, upper], offsets=[0, -1, 1], shape=(Nx+1, Nx+1), format='csr')

        u = np.zeros((Nt, Nx + 1))
        # Set initial condition
        u[0, :Nx] = f_init
        #run over time steps
        for i in tqdm(range(1, Nt), f'simulating {Nt} time steps'):
            b = u[i-1]
            b[0] = b[-1] = 0.0 # boundary conditions
            res = A.dot(u[i-1])
            u[i] = res
        end = time.time()

        cell_Reynolds = c/s
        print(f'Cell Reynolds Number: {cell_Reynolds:.3f}')
        if cell_Reynolds > 1.0 or cell_Reynolds < 0.1:
            print(f"Stability with may not be guaranteed with Cell Reynolds Number {cell_Reynolds:.3f}")

    else:

        u = np.zeros((Nt, Nx + 1))
        # Set initial condition
        u[0, :Nx] = f_init


        # run over time steps
        for i in tqdm(range(1, Nt), f'simulating {Nt} time steps'):
            # Representation of sparse matrix and right-hand side
            main = np.zeros(Nx + 1)
            lower = np.zeros(Nx)
            upper = np.zeros(Nx)
            b = np.zeros(Nx + 1)

            # discretization for Lax-Wendroff (forward?)
            c = a[i] * dt / dx
            s = alpha[i] * dt / dx ** 2
            r1 = 0.5 * (2 * s + c + c ** 2)
            r2 = 1 - 2 * s - c ** 2
            r3 = 0.5 * (2 * s - c + c ** 2)

            # Precompute sparse matrix
            main[:] = r2
            lower[:] = r1
            upper[:] = r3
            # Insert boundary conditions
            main[0] = 1
            main[Nx] = 1
            A = scipy.sparse.diags(diagonals=[main, lower, upper], offsets=[0, -1, 1], shape=(Nx + 1, Nx + 1), format='csr')

            b = u[i - 1]
            b[0] = b[-1] = 0.0  # boundary conditions
            res = A.dot(u[i - 1])
            u[i] = res

        end = time.time()
        cell_Reynolds = c / s
        R_c_max, R_c_min = cell_Reynolds.max(), cell_Reynolds.min()
        print(f'Cell Reynolds Number range: {R_c_min:.3f}, {R_c_min:.3f}')
        if R_c_max > 1.0 or R_c_min < 0.1:
            print(f"Stability with may not be guaranteed with Cell Reynolds Number outised range (0.1, 1.0)")
        u = u[:, :-1]
elif solver=='Hu-2021':
    u = np.zeros((Nt, Nx))
    # Set initial condition
    u[0] = f_init
    r = np.zeros(Nt)

    if not isinstance(alpha, np.ndarray):
        alpha = alpha * np.ones(Nt)
    if not isinstance(a, np.ndarray):
        a = a * np.ones(Nt)

    c_count = 0

    # run over time steps
    for i in tqdm(range(1, Nt), f'simulating {Nt} time steps'):
        if alpha[i] > 0.001:
            # Representation of sparse matrix and right-hand side
            main = np.zeros(Nx)
            lower = np.zeros(Nx - 1)
            upper = np.zeros(Nx - 1)

            # discretization for Hu-2021
            c = alpha[i] * dt / dx # alpha[i] * dt / dx
            critval = 1.5*(a[i]/5)
            if c < critval:
                c = critval
                alpha[i] = c * dx / dt
                if c_count < 1:
                    c_count += 1
                    print(f'resetting alpha to {alpha[i]:.5f} to achieve numerical stability')

            Ms = M(x, a[i], alpha[i], x_L=x_rest)

            if len(np.where(Ms == np.inf)[0]) > 0:
                raise ValueError('Infinite flux detected!')

            M_harm = np.zeros(Nx-1)  # idk
            for j in range(Nx-1):
                M_harm[j] = harm_mean(Ms[j], Ms[j+1])


            r1 = -c*M_harm[:-1]/Ms[:-2]
            r2 = 1+c*(M_harm[1:] + M_harm[:-1])/Ms[1:-1]
            r3 = -c*M_harm[1:]/Ms[2:]

            # Precompute sparse matrix with boundary conditions
            main[1:-1] = r2
            lower[:-1] = r1
            upper[1:] = r3
            # von Neumann Boundary conditions?
            # main[0] = main[1]
            # main[-1] = main[-2]
            main[0] = 1
            main[-1] = 1
            A = scipy.sparse.diags(diagonals=[main, lower, upper], offsets=[0, -1, 1], shape=(Nx, Nx), format='csr')

            b = u[i - 1]
            b[0] = b[-1] = 0.0  # boundary conditions

            # firing rate / outgoing flux
            # if i > 1:
            #     r[i] = (u[i-2].sum() - u[i-1].sum())
            r[i] = (alpha[i] / dx) * (u[i - 1, -2]) + ((x[-2] - x_rest) + a[i]) * u[i - 1, x_thr_idx]


            u[i] = scipy.sparse.linalg.spsolve(A, b)
            J_out = r[i] * (np.heaviside((x + dx) - x_reset, 1) - np.heaviside((x - dx) - x_reset, 1))  # / (1+alpha[i])**2
            if J_out.max() != 0:
                J_out /= J_out.sum()
                J_out *= (1-u[i].sum())
            # J_out *= -SG(x+dx, x_L=x_rest, a=a[i], dx=dx, alpha=alpha[i])
            if J_out.sum() > 1e-2:
                v = 1
            u[i] += J_out

    end = time.time()
print(f'grid coeff c: {c_out:.4f}')
print(f"computation time: {end-start:2f}s")
print(f"initial volume: {u[0].sum():.2f}")
print(f"Part of initial volume left: {u[-1].sum():.2f}")
def plot_sol(u, t, x, alpha=1):
    u_plot = u[:].T
    fig = plt.figure(figsize=(10, 4.25))
    x_mesh, y_mesh = np.meshgrid(t, x)
    ax = fig.add_subplot(1, 2, 1)
    z_min, z_max = u.min(), min(u[-1].mean()*10, u.max())
    c = ax.pcolormesh(x_mesh, y_mesh, u_plot, cmap='viridis', vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax)
    if isinstance(alpha, (float, int)) and isinstance(a, (float, int)):
        ax.set_title(f"Time evolution of the diffusion equation for alpha = {alpha:.2f} and a = {a:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax = fig.add_subplot(1, 2, 2)
    a_plot = a /a.max() * r.max()
    ax.plot(t, r)
    ax.plot(t, a_plot)
    ax.legend(['firing rate (AU)', 'input current (AU)'])
    ax.set_xlabel('t')
    plt.tight_layout()
    plt.show()

plot_sol(u=u, t=t, x=x_, alpha=alpha)

# convert mean of stationary to voltage range
end_mean_idx = np.argmax(u[-1])
end_mean = x_[end_mean_idx]
print(f'mean of stationary solution = {end_mean:.2f}')
print(f'x_rest = {x_rest_:.2f}')

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(x_, u[-1])
ax.plot(x_, u[0], color='k', alpha=0.6)
ax.set_xlabel('x')
ax.set_xlabel('u(x)')
ax.set_title(f'distribution at t= {T}')
ax.grid()
ax.vlines(x=x_rest_, ymax=u[-1].max(), ymin=u[-1].min(), linestyle='--', color='k')
ax.text(x_rest + 0.03, 0.03, 'x_rest', fontsize=12, transform=ax.transAxes)
ax.text(x0_new - 0.1, 0.93, 'u_0', fontsize=12, transform=ax.transAxes)
ax.set_ylim((u[-1].min(), u[-1].max()))
plt.show()
