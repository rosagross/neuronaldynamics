import numpy as np
from scipy.stats import norm
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('TkAgg')

solver = 'Hu-2021' # 'LW', 'Hu-2021'

def M(x, a, alpha, x_L=-5):
    """
    compute Scharfetter-Gummel flux functional
    :param x: x
    :param a: drift
    :param alpha: diffusion
    :param x_L: resting point
    :return: M: Scharfetter-Gummel flux functional
    """
    U = ((1/2)*x - x_L - a)*x/alpha
    return np.exp(-U)

def harm_mean(x1, x2):
    return 2/((1/x1)+(1/x2))

dx = 0.1
dt = 0.01
T = 10
x0 = -4
sigma0 = 1.0
alpha = 0.2
a = 1
x = np.arange(-10, 10, dx)
t = np.arange(0, T, dt)

f_init = norm.pdf(x, x0, sigma0)
f_init[0] = f_init[-1] = 0
f_init /= f_init.sum()
Nx = x.shape[0]
Nt = t.shape[0]

a = a * np.ones(Nt) + 1.5*np.sin(t)

start = time.time()
if solver == 'LW':
    if not (isinstance(a, np.ndarray) and not isinstance(alpha, np.ndarray)):

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

elif solver=='Hu-2021':
    u = np.zeros((Nt, Nx + 1))
    # Set initial condition
    u[0, :Nx] = f_init

    if not isinstance(alpha, np.ndarray):
        alpha = alpha * np.ones(Nt)


    # run over time steps
    for i in tqdm(range(1, Nt), f'simulating {Nt} time steps'):
        # Representation of sparse matrix and right-hand side
        main = np.zeros(Nx - 1)
        lower = np.zeros(Nx - 2)
        upper = np.zeros(Nx - 2)

        # discretization for Hu-2021
        c = alpha[i] * dt / dx**2  # alpha[i] * dt / dx

        Ms = M(x, a[i], alpha[i])
        M_harm = np.zeros(Nx-1)  # idk
        for j in range(Nx-1):
            M_harm[j] = harm_mean(Ms[j], Ms[j+1])


        r1 = -c*M_harm[:-2]/Ms[:-3]
        r2 = 1+c*(M_harm[1:-1]/Ms[1:-2] + M_harm[:-2]/Ms[:-3])
        r3 = -c*M_harm[1:-1]/Ms[1:-2]

        # Precompute sparse matrix
        main[1:-1] = r2
        lower[:-1] = r1
        upper[1:] = r3

        # Insert boundary conditions
        main[0] = 1
        main[Nx-2] = 1
        A = scipy.sparse.diags(diagonals=[main, lower, upper], offsets=[0, -1, 1], shape=(Nx - 1, Nx - 1), format='csr')

        b = u[i - 1, 1:-1]
        # b[0] = b[-1] = 0.0  # boundary conditions
        u[i, 1:-1] = scipy.sparse.linalg.spsolve(A, b)

    end = time.time()

print(f"computation time: {end-start:2f}s")
print(f"Part of initial volume left: {u[-1].sum():.2f}")
def plot_sol(u, t, x, alpha=1):
    u_plot = u[:, :-1].T
    fig = plt.figure(figsize=(10, 4.25))
    x_mesh, y_mesh = np.meshgrid(t, x)
    ax = fig.add_subplot(1, 1, 1)
    z_min, z_max = u.min(), u.max()
    c = ax.pcolormesh(x_mesh, y_mesh, u_plot, cmap='viridis', vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax)
    if isinstance(alpha, (float, int)) and isinstance(a, (float, int)):
        ax.set_title(f"Time evolution of the diffusion equation for alpha = {alpha:.2f} and a = {a:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.tight_layout()
    plt.show()

plot_sol(u=u, t=t, x=x, alpha=alpha)