import numpy as np
import matplotlib
matplotlib.use('TkAgg')

def RosenbrockND(x):
    """
    N dimensional Rosenbrock function
    for 3D has minimum at (1, 1, 1,)
    :param x: (np.ndarray) input array
    :return: out_sum (np.float64) output value
    """
    assert len(x.shape) > 1, "Rosenbrock Function needs at minimum a 1x1 array as input"
    x_shape = x.shape
    dim, n = x_shape[:2]
    out = np.zeros(x_shape)
    for j in range(n):
        for i in range(dim-1):
            out[i] = 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    out_sum = np.sum(out, axis=0)
    return out_sum