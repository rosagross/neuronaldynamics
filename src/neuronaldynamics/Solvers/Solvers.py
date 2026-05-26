import scipy as sc
import numpy as np


def Euler(f, x0, t, step_size=0.1):
    """ explicit Euler method
    :parameter
    f: function, function(x, t) of x and t
    x0: np.ndarray, initial value for x
    t: np.ndarray, array of time points that are evaluated. They need to be separated by a homogeneous step_size
    step_size: float, step size_ between consecutive time points in t
    :return
    x: np.ndarray, solution
    """
    dim = x0.shape[0]
    duration = t.shape[0]
    x = np.zeros([dim, duration])
    x[:, 0] = x0

    for i in range(0, len(t) - 1):
        x[:, i + 1] = x[:, i] + step_size * f(x[:, i], t[i])

    return x