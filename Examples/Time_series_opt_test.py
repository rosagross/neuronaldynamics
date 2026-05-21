import numpy as np
import matplotlib
from matplotlib.lines import lineStyles

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from Optimizers.Testfunctions import RosenbrockND
from Optimizers.Optimizer import Hierarchical_Random, GA

def main():

    class Sine_test_function():
        """
        Rosenbroch function version in 2D
        in 2D should have minimum at (1, 1)
        """
        def __init__(self, parameters={}):
            self.x1 = np.array([[1]])
            self.x2 = np.array([[1]])
            self.t = np.linspace(0, 10, int(1e3))
            self.__dict__.update(parameters)
        def simulate(self):
            self.y = self.x1*np.sin(self.x2*self.t)

    t = np.linspace(0, 10, int(1e3))
    ref = 3.1*np.sin(0.3*t)

    def test_hierarchical_random():
        testfunction = Sine_test_function()
        model_parameters = ['x1', 'x2']
        opt_params = {}
        opt_params['model_parameters'] = model_parameters
        opt_params['y'] = ref
        opt_params['simulation_class'] = testfunction
        opt_params['simulate'] = testfunction.simulate
        opt_params['bounds'] = [[0, 10], [0, 10]]
        opt_params['x_out'] = 'y'
        opt_params['n_grid'] = 1000
        opt_params['tolerance'] = 0.01
        optimizer1 = Hierarchical_Random(parameters=opt_params)
        optimizer1.run()
        optimum = optimizer1.optimum
        print(optimum)

    testfunction = Sine_test_function()
    model_parameters = ['x1', 'x2']
    opt_params = {}
    opt_params['model_parameters'] = model_parameters
    opt_params['serial_computation'] = True
    opt_params['simulation_class'] = testfunction
    opt_params['simulate'] = testfunction.simulate
    opt_params['bounds'] = [[0, 10], [0, 10]]
    opt_params['x_out'] = 'y'
    opt_params['reference'] = ref
    opt_params['n_iter'] = 50
    opt_params['N1'] = 100
    opt_params['N2'] = 100
    opt_params['tolerance'] = 1e-1
    opt_params['single_run_tol'] = 1e-2
    optimizer = GA(parameters=opt_params)
    # opt_params['max_iter'] = 1000
    # optimizer = Hierarchical_Random(parameters=opt_params)
    optimizer.run()
    optimal_param = optimizer.optimum
    # optimizer.plot_fit()
    # ps = optimizer.parameter_evolution
    opt_sine = Sine_test_function({'x1': optimal_param[0], 'x2': optimal_param[1]})
    opt_sine.simulate()
    y_opt = opt_sine.y
    plt.plot(t, y_opt, label='optimum', linestyle='--')
    plt.plot(t, ref, label='reference')
    plt.show()

if __name__ == "__main__":
    main()