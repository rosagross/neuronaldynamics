import numpy as np
import time
import h5py
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def get_stability_2D(eigvals):
    """function to test the stability of a 2D system by evaluating common cases of its eigenvalues """
    stability = 'untested'
    lambda1, lambda2 = eigvals
    if lambda1.real == 0 or lambda2.real == 0:
        stability = 'unknown (Eigenvalue zero)'
    elif lambda1.imag == 0 and lambda1.imag == 0:
        if lambda1 < 0 and lambda2 < 0:
            stability = 'stable node'
        elif lambda1 > 0 and lambda2 > 0:
            stability = 'unstable node'
    else:
        if lambda1 < 0 and lambda2 < 0:
            stability = 'stable spiral'
        elif lambda1 > 0 and lambda2 > 0:
            stability = 'unstable spiral'

    if np.sign(lambda1.real) != np.sign(lambda2.real):
        stability = 'unstable saddle node'

    return stability

def nrmse(reference, x):
    """
    Caclucaltes the nrmse, Normalized Root Mean Squared Error between a reference and an array, x
    :param reference: np.ndarray
        reference to test the input against
    :param x: np.ndarray
        input array to test against reference
    :return: nrmse: np.ndarray
        nrmse
    """
    diff = np.subtract(x, reference)
    square = np.square(diff)
    mse = square.mean()
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(x) - np.min(x))
    return nrmse


def raster(event_times_list, color='k'):
  """
  Creates a raster plot **with spikes saved at 300 dpi as raster art**

  Original code from https://gist.github.com/kylerbrown/5530238

  Parameters
  ----------
  event_times_list : iterable
                     a list of event time iterables
  color : string
          color of vlines

  Returns
  -------
  ax : an axis containing the raster plot

  This version attempts to rasterize the plot
  Reference: https://matplotlib.org/stable/gallery/misc/rasterization_demo.html
  """
  ax = plt.gca()
  for ith, trial in enumerate(event_times_list):
    plt.vlines(trial, ith + .5, ith + 1.5, color=color, rasterized=True)
  plt.ylim(.5, len(event_times_list) + .5)
  return ax


def time_bin(x, bin_size=5):
    """
    function that creates sum over bin region and then replaces all entries in this region with its sum
    :param x: array, np.1darray
    :param bin_size: bin_size, int
    :return: binned array
    """

    # Create an array of indices for the chunks
    indices = np.arange(len(x)) // bin_size
    #Sum the values in each chunk using bincount
    sums = np.bincount(indices, weights=x)
    # Map the sums back to the original array
    result = sums[indices] / bin_size

    return result

# Custom formatter function to divide labels by 10
def make_div_func(value, set_int):
    """
     function that divides x by value used for formatting axes in matplotlib plots
    :param value: value by which x is divided
    :param set_int: bool, if true, function returns ints
    :return: function that does x/value
    """
    def function(x, pos):
        """
        subfunction of make_div_func since matplotlib.ticker.FuncFormatter() needs a function with arguments
        x and pos as input
        :param x: incoming value
        :param pos: parameter for matplotlib.ticker.FuncFormatter()
        :return: string of function x/value
        """
        if set_int:
            return f'{int(x / value)}'
        else:
            return f'{x / value}'
    return function

def divide_axis(ax, value=10, axis='x', set_int=False):

    function = make_div_func(value, set_int)

    if axis == 'x':
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(function))
    elif axis == 'y':
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(function))
    else:
        raise NotImplementedError('only "x" or "y" are implemented as axis for this function')



# Function to find the closest value in the reference array
def find_closest(array, value):
    """
    function that finds closest idx of closest point in array to a given value
    :param array: array
    :param value: value
    :return: idx
    """
    return np.abs(array - value).argmin()

# Map each float to the closest integer in reference_array
def round_to_1dgrid(x, grid, idx=False):
    """
    function that rounds input data x to closest points on a 1d grid
    :param x: input array
    :param grid: grid
    :return: mapped array
    """
    if not type(x) == np.ndarray:
        x = np.array([x])

    idxs = [find_closest(grid, val) for val in x]
    res = np.array(idxs)
    if idx:
        return res, np.array(idxs, dtype=int)
    else:
        return res

def record_time(function):
    """
    Function the times a code snippet
    :param function:
    :return: time it took
    """

    start = time.time()
    exec(function)
    end = time.time()
    return end-start

def list_flatten(in_list):
    """
    function that flattens a list of lists
    param:
    in_list: list
            input list
    """
    return [k for l in in_list for k in l]

def get_combintations(array1, array2):
    """
    Function that computes all 2 element combinations of two 1D arrays
    :param array1: np.ndarray
                input array 1
    :param array2: np.ndarray
                input array 2
    :return: grid_combinations: np.ndarray
                output array of 2 element combinations
    """
    # Reshape the arrays to be 2D and compatible for broadcasting
    array1_reshaped = array1[:, np.newaxis]
    array2_reshaped = array2[np.newaxis, :]

    # Broadcast arrays and create a grid of combinations
    grid_combinations = np.array(np.meshgrid(array1, array2)).T.reshape(-1, 2)
    return grid_combinations[0]


def compare_solution(sol_1, sol_2, x=None, save_fname=None):
    """
    Function that plots a 2D solution against a reference solution and their nrmse
    :param sol_1: np.ndarray
        Solution 1 (reference)
    :param sol_2: np.ndarray
        Solution 2 (test)
    :param x: np.ndarray
        x values (optional) default: None
    """

    if not type(x) == np.ndarray:
        x = np.arange(sol_1.shape[0])
    error = nrmse(sol_1, sol_2)
    diff = np.abs(sol_1 - sol_2)
    fig = plt.figure(figsize=(10, 4.25))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, sol_1)
    ax1.plot(x, sol_2)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, diff)
    ax2.set_title(f"Difference | nrmse: {error:.4f}")
    ax1.set_label(['Nykamp', 'LIF'])
    ax1.set_xlabel('t in ms')
    ax2.set_xlabel('t in ms')
    ax1.set_title('Firing rates in Hz')
    if save_fname==None:
        plt.show()
    else:
        plt.savefig(save_fname)

def compare_firing_rate(fname1, fname2, idx=0, n_neurons=1000, dt=0.1, smooth = True, save_fname=None):

    with h5py.File(fname1 + '.hdf5', 'r') as h5file:
        r1 = np.array(h5file['r'])
        t = np.array(h5file['t'])
    with h5py.File(fname2 + '.hdf5', 'r') as h5file:
        r2 = np.array(h5file['r'])

    r_compare_2 = r2[idx]*(1/dt)*(1000/n_neurons)
    r_compare_1 = r1[idx] * 1000
    if smooth:
        r_compare_2 = gaussian_filter1d(r_compare_2, sigma=10)
    compare_solution(r_compare_1, r_compare_2, x=t, save_fname=save_fname)


