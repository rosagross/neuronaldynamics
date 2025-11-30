# -------------------------
# Hu-2021 / Chang-Cooper style solver (replacement for old block)
# -------------------------
import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import exp
dx = 0.1
dt = 0.01
T = 10
x0 = -4
sigma0 = 1.0
alpha = 0.01
x = np.arange(-10, 10, dx)
t = np.arange(0, T, dt)

f_init = scipy.stats.norm.pdf(x, x0, sigma0)
f_init[0] = f_init[-1] = 0
f_init /= f_init.sum()
Nx = x.shape[0]
Nt = t.shape[0]

a = 5*(np.ones(Nt) + np.sin(t))

cell_wise = False

def B_face_from_a(a_val, x, x_rest=-5):  # example: constant drift a_val; replace if drift depends on v
    # return array of face velocities B_{i+1/2}, length = Nx+1 (faces)
    # We'll use simple constant advection `a_val` here; if you have B(v,t), evaluate at faces
    res = np.zeros(Nx + 1)
    # res[:-1] = -(x-x_rest) + a_val
    res[:] = a_val
    return res


def B_face_from_field(B_node):
    # If B given at nodes, compute face values by averaging
    return 0.5 * (np.concatenate(([B_node[0]], B_node)) + np.concatenate((B_node, [B_node[-1]])))


def SG(alpha_face):
    # helper: stable eval of alpha/(exp(alpha)-1)
    if abs(alpha_face) < 1e-8:
        return 1.0 - alpha_face / 2.0 + alpha_face ** 2 / 12.0  # taylor expansion?
    return alpha_face / (np.exp(alpha_face) - 1.0)

def SG_v(alpha_face):
    return alpha_face / (np.exp(alpha_face) - 1.0 +1e-6)

# grid / storage
u = np.zeros((Nt, Nx))
u[0, :] = f_init.copy()
# ensure proper normalization (with dx)
u[0, :] /= (u[0, :].sum() * dx)

# choose reset index for V_reset (here assume reset V_r = some x value near 0)
V_reset = -5.0
k_reset = np.argmin(np.abs(x - V_reset))

# Time-stepping (implicit time step for diffusion via linear solve)
for n in tqdm(range(1, Nt), desc='Hu-style time stepping'):
    # current coefficients
    alpha_n = alpha if not isinstance(alpha, np.ndarray) else alpha[n]
    a_n = a if not isinstance(a, np.ndarray) else a[n]

    # Build face quantities
    # Drift (face velocities) - replace with your physical B(v,t) if not constant
    B_face = B_face_from_a(a_n, x)  # length Nx+1

    # Diffusion coefficient (assumed scalar here); if function of v use array at faces
    D = alpha_n

    # compute Scharfetter-Gummel interface flux coefficients
    # alpha_face = B_face * dx / D
    # SG flux J_{i+1/2} = (D/dx) * ( B(alpha_face) * u_{i+1} - B(-alpha_face) * u_i )
    # where B(s) = s/(exp(s)-1)
    # We'll assemble linear system for implicit Euler: u^{n+1} - dt*(-div J^{n+1}) = u^n
    # This yields tridiagonal matrix A to solve: A * u^{n+1} = u^n + source_from_reset

    Nx_inner = Nx  # number of unknowns (cells)
    # prepare tridiagonal entries
    diag = np.zeros(Nx_inner)
    lower = np.zeros(Nx_inner - 1)
    upper = np.zeros(Nx_inner - 1)

    if cell_wise:
    # faces: indices 0..Nx (face 0 at left boundary, face Nx at right boundary)
    # For interior faces i = 1..Nx-1, they connect cell i-1 and i
        for i_cell in range(Nx_inner):
            # compute contributions from left face (i-1/2) and right face (i+1/2)
            # left face index:
            iL = i_cell  # face between cell i_cell-1 and i_cell is face index i_cell
            iR = i_cell + 1  # face between cell i_cell and i_cell+1 is face index i_cell+1

            # Left face flux linearization coefficients:
            # define alpha_f = B_face[iL]*dx / D  (watch D==0 case below)
            if D <= 0:
                # purely hyperbolic: fall back to upwind (simple)
                # We'll not rely on implicit solve. Raise or handle separately.
                raise ValueError("D <= 0: use hyperbolic upwind solver instead (a=0 limit).")
            alphaL = B_face[iL] * dx / D
            alphaR = B_face[iR] * dx / D

            BL = SG(alphaL)
            BmL = SG(-alphaL)
            BR = SG(alphaR)
            BmR = SG(-alphaR)

            # flux linear maps:
            # J_{i-1/2} = (D/dx) * ( BL * u_i - BmL * u_{i-1} )   (note sign depends on indexing)
            # J_{i+1/2} = (D/dx) * ( BR * u_{i+1} - BmR * u_i )

            # divergence contribution for cell i:
            # - (J_{i+1/2} - J_{i-1/2}) / dx = combination of u_{i-1}, u_i, u_{i+1}
            coef_left = (D / dx) * BmL / dx  # multiplies u_{i-1}
            coef_center = - (D / dx) * (BL + BmR) / dx  # multiplies u_i
            coef_right = (D / dx) * BR / dx  # multiplies u_{i+1}

            # multiply by dt because implicit Euler: u^{n+1} - dt * div J^{n+1} = u^n
            diag[i_cell] = 1.0 - dt * coef_center
            if i_cell > 0:
                lower[i_cell - 1] = - dt * coef_left
            if i_cell < Nx_inner - 1:
                upper[i_cell] = - dt * coef_right
    else:
        c = dt * D/dx**2
        a_of_x = np.ones(Nx+1) * a[n]
        # diag[:] = 1.0 - dt/D*(a_of_x[:-1]*(np.exp(-a_of_x[:-1]/alpha*dx) - 1.0) - a_of_x[0:-1] * np.exp(-a_of_x[0:-1] / alpha * dx) - 1.0)
        # lower[1:] = -dt/D*a_of_x[:-1]*(np.exp(-a_of_x[:-1]/alpha*dx) - 1.0)
        # upper[:-1] = dt/D*a_of_x[0:-1] * (np.exp(a_of_x[0:-1] / alpha * dx) - 1.0)

        lower[:] = -c * SG_v(-dx/D*a_of_x[:-2])
        upper[:] = -c * SG_v(dx / D * a_of_x[1:-1])
        diag[:] = 1+c*(SG_v(-dx/D*a_of_x[:-1]) + SG_v(dx / D * a_of_x[1:]))

    # Boundary handling (left boundary no-flux: set flux at face 0 = 0 => modifies first eq)
    # For face 0, J_{-1/2} is not present; scheme above implicitly assumed internal faces.
    # For simplicity enforce zero flux at left by modifying first row (Dirichlet-like or natural)
    # Already the above loop used faces iL, iR; ensure first row is correctly consistent
    # (optionally you can replace diag[0], lower[0] etc to implement exact BC).
    # For right boundary, there is outgoing flux at face Nx; compute outgoing flux after solve
    # and re-inject into reset cell via source term on RHS:
    RHS = u[n - 1, :].copy()  # RHS = u^n

    # Solve tridiagonal (use sparse)
    A = scipy.sparse.diags([diag, lower, upper], offsets=[0, -1, 1], shape=(Nx_inner, Nx_inner), format='csr')
    u_new = scipy.sparse.linalg.spsolve(A, RHS)

    # Now compute outgoing flux at right face (explicit in u_new)
    # J_{N+1/2} at face index Nx:
    alpha_face_right = B_face[Nx] * dx / D
    Br = SG(alpha_face_right)
    Bmr = SG(-alpha_face_right)
    # J_right = (D/dx) * ( Br * u_{N} - Bmr * u_{N-1} )
    # Beware indexing: u_new has indices 0..Nx-1 ; u_{N} doesn't exist (outside). For threshold we often set p(V_F)=0.
    # If we assume cell at threshold has p = 0, outgoing flux approx:
    J_out = -(D / dx) * (Br * 0.0 - Bmr * u_new[-2])

    # Re-inject mass into reset cell
    u_new[k_reset] += (dt / dx) * J_out  # add mass (dt*flux is mass), normalized by cell volume dx

    # store
    u[n, :] = u_new

# end time loop
# final mass check
mass = u[-1, :].sum() * dx
print("final mass:", mass)

def plot_sol(u, t, x, alpha=1.0):
    u_plot = u[:, :].T
    fig = plt.figure(figsize=(10, 4.25))
    x_mesh, y_mesh = np.meshgrid(t, x)
    ax = fig.add_subplot(1, 1, 1)
    z_min, z_max = u.min(), u.max()
    c = ax.pcolormesh(x_mesh, y_mesh, u_plot, cmap='viridis', vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax)
    if isinstance(alpha, (float, int)) and isinstance(a, (float, int)):
        ax.set_title(f"Time evolution of the non-linear advection-diffusion \n"
                     f"equation for alpha = {alpha:.2f} and a = {a:.2f}")
    else:
        ax.set_title(f"Time evolution of the non-linear advection-diffusion \n"
                     f"equation for time varying coefficients")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.tight_layout()
    plt.show()

plot_sol(u=u, t=t, x=x, alpha=alpha)
