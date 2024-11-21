import numpy as np
import matplotlib.pyplot as plt

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


def time_bin(x, bin=5):
    """
    function that creates sum over bin region and then replaces all entries in this region with its sum
    :param x: array, np.1darray
    :param bin: bin, int
    :return: binned array
    """

    # Create an array of indices for the chunks
    indices = np.arange(len(x)) // bin
    #Sum the values in each chunk using bincount
    sums = np.bincount(indices, weights=x)
    # Map the sums back to the original array
    result = sums[indices]

    return result


