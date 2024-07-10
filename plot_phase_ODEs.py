# dynamical system phase space plot

import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import odeint

plt.rcParams['text.usetex'] = True


# def system(vect, t):
#     x, y = vect
#     return [x - y - x * (x ** 2 + 5 * y ** 2), x + y - y * (x ** 2 + y ** 2)]

def system(vect, t):
    x, y = vect
    return [-5*(x-2)**3 + (x-2)**2 + 5*(x-2) + 2 -y, 0.015*x**8-y]

def system2(vect, t):
    x, y = vect
    return [-5*(x-2)**3 + (x-2)**2 + 5*(x-2) + 2 -y, 0.015*(x-0.5)**8-y +0.5]

x = np.linspace(0, 3, 20)
y = np.linspace(0, 5, 10)
# xx, yy = np.meshgrid(x, y)
vect0 = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
t = np.linspace(0, 10, int(2e3))

plot = plt.figure()

for i in range(vect0.shape[0]):
    v = [vect0[i][0], vect0[i][1]]
    sol = odeint(system2, v, t)
    # plt.plot(t, sol[:, 0])
    # plt.plot(t, sol[:, 1])
    # plt.show()
    # plt.close()
    plt.quiver(sol[:-1, 0], sol[:-1, 1], sol[1:, 0] - sol[:-1, 0], sol[1:, 1] - sol[:-1, 1], scale_units='xy',
               angles='xy', scale=1, color='k')
    plt.plot(sol[:, 0], sol[:, 1], color='k', alpha=0.2)

# plot nullclines
x0 = np.arange(0, 5, 0.01)
b = 0.5
f1 = -5*(x0-2)**3 + (x0-2)**2 + 5*(x0-2) + 2
f2 = 0.015*(x0-b)**8 + b
plt.plot(x0, f1, c='orange')
plt.plot(x0, f2, c='blue')

plt.xlim(0, 3)
plt.ylim(-0.1, 5)
plt.xlabel(r'V')
plt.ylabel(r'n')
plt.title(r'I_{Na, p} \quad + \quad I_K \textit{model}')

idx = np.argwhere(np.diff(np.sign(f1 - f2))).flatten()
plt.plot(x0[idx], f1[idx], 'ro')


plt.show()