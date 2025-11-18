import numpy as np
from scipy.stats import norm
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('TkAgg')

dx = 0.1
dt = 0.01
T = 10
x0 = 2
sigma0 = 1
alpha = 0.5
x = np.arange(-10, 10, dx)
t = np.arange(0, T, dt)

f_init = norm.pdf(x, x0, sigma0)
f_init[0] = f_init[-1] = 0
f_init /= f_init.sum()
Nx = x.shape[0]
Nt = t.shape[0]
F = (alpha/dx**2*dt)

start = time.time()
# Representation of sparse matrix and right-hand side
main = np.zeros(Nx+1)
lower = np.zeros(Nx)
upper = np.zeros(Nx)
b = np.zeros(Nx+1)

# Precompute sparse matrix
main[:] = 1 + 2*F
lower[:] = -F
upper[:] = -F
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
    u[i] = scipy.sparse.linalg.spsolve(A, b)

end = time.time()
print(f'Fourier Number: {F:.3f}')
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
    ax.set_title(f"Time evolution of the diffusion equation for alpha ={alpha}")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.tight_layout()
    plt.show()

plot_sol(u=u, t=t, x=x, alpha=alpha)