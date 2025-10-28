import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from Optimizers.Testfunctions import RosenbrockND
from Optimizers.Optimizer import Hierarchical_Random, GA
def plot_Rosenbrock():
    X, Y = np.mgrid[-2:2.1:0.01, -1:3.1:0.01]
    xs = np.array((X, Y))
    y = RosenbrockND(xs)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    z_min, z_max = y.min(), y.max()
    c = ax.pcolormesh(X, Y, y, cmap='jet', vmin=z_min, vmax=z_max)
    ax.scatter(1, 1, marker='x', s=20, c='red')
    y_min = np.argmin(y)
    y_flat = y.flatten()
    y_min_val = y_flat[y_min]
    print(f'Min value: {y_min_val}')
    # a = np.where(y==0)
    fig.colorbar(c, ax=ax)
    plt.show()

class Rosenbrock_2D():
    """
    Rosenbroch function version in 2D
    in 2D should have minimum at (1, 1)
    """
    def __init__(self, parameters={}):
        self.x1 = np.array([[1]])
        self.x2 = np.array([[1]])
        self.__dict__.update(parameters)
    def simulate(self):
        x = np.array((self.x1, self.x2))
        if len(x.shape) < 2:
            x = x[:, np.newaxis]
        self.y = RosenbrockND(x)

def test_hierarchical_random():
    testfunction = Rosenbrock_2D()
    model_parameters = ['x1', 'x2']
    opt_params = {}
    opt_params['model_parameters'] = model_parameters
    opt_params['y'] = np.array([0])
    opt_params['simulation_class'] = testfunction
    opt_params['simulate'] = testfunction.simulate
    opt_params['bounds'] = [[-2, 1], [-1, 3]]
    opt_params['x_out'] = 'y'
    opt_params['n_grid'] = 1000
    opt_params['tolerance'] = 0.01
    optimizer1 = Hierarchical_Random(parameters=opt_params)
    optimizer1.run()
    optimum = optimizer1.optimum
    print(optimum)

testfunction = Rosenbrock_2D()
model_parameters = ['x1', 'x2']
opt_params = {}
opt_params['model_parameters'] = model_parameters
opt_params['y'] = np.array([0])
opt_params['simulation_class'] = testfunction
opt_params['simulate'] = testfunction.simulate
opt_params['bounds'] = [[-2, 1], [-1, 3]]
opt_params['x_out'] = 'y'
opt_params['reference'] = 0
opt_params['n_iter'] = 20
opt_params['N1'] = 20
opt_params['N1'] = 20
opt_params['tolerance'] = 1e18
optimizer = GA(parameters=opt_params)
# opt_params['max_iter'] = 1000
# optimizer = Hierarchical_Random(parameters=opt_params)
optimizer.run()
optimal_param = optimizer.optimum
# optimizer.plot_fit()
ps = optimizer.parameter_evolution
opt_Rosenbrock = Rosenbrock_2D({'x1': optimal_param[0], 'x2': optimal_param[1]})
opt_Rosenbrock.simulate()
y_opt = opt_Rosenbrock.y
print(f'y_opt = {y_opt}')

def animate_sol(params):
    X, Y = np.mgrid[-2:2.1:0.01, -1:3.1:0.01]
    xs = np.array((X, Y))
    y = RosenbrockND(xs)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    z_min, z_max = y.min(), y.max()
    c = ax.pcolormesh(X, Y, y, cmap='jet', vmin=z_min, vmax=z_max)
    sc = ax.scatter(0, 0, marker='x', s=20, c='white')
    ax.scatter(1, 1, marker='x', s=20, c='red')
    y_min = np.argmin(y)
    y_flat = y.flatten()
    y_min_val = y_flat[y_min]
    print(f'Min value: {y_min_val}')
    # a = np.where(y==0)
    fig.colorbar(c, ax=ax)

    def update(frame):
        xs = params[frame]
        sc.set_offsets(xs)
        return (sc,)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=params.shape[0], interval=1)
    # plt.show()
    ani.save(filename='rosenbrock.gif', writer="pillow")
    plt.close()


animate_sol(np.array(ps))
