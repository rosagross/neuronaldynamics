import numpy as np
import time
import re
import h5py
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
from scipy.ndimage import gaussian_filter1d, shift
from scipy.signal import correlate, find_peaks
matplotlib.use('TkAgg')

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
    nrmse = rmse / (np.max(x) - np.min(x) + 1e-12)
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


def compare_solution(sol_1, sol_2, x=None, save_fname=None, titles = ['plot1', 'Plot2']):
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
    ax1.legend(titles)
    if save_fname==None:
        plt.show()
    else:
        plt.savefig(save_fname)


def plot_rates(solutions, x=None, save_fname=None, titles = ['Plot1', 'Plot2']):
    """
    Function that plots a 2D solution against a reference solution and their nrmse
    :param solutions: np.ndarray
        Solutions
    :param x: np.ndarray
        x values (optional) default: None
    :param titles: list of strings
        titles of the plotted rates default: ['Plot1', 'Plot2']
    """

    if not type(x) == np.ndarray:
        x = np.arange(solutions.shape[1])

    fig = plt.figure(figsize=(10, 4.25))
    n_plots = solutions.shape[0]
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(n_plots):
        ax1.plot(x, solutions[i])

    ax1.set_xlabel('t in ms')
    ax1.set_title('Firing rates in Hz')
    ax1.legend(titles)
    if save_fname==None:
        plt.show()
    else:
        plt.savefig(save_fname)

def compare_firing_rate(fname1, fname2, idx=0, n_neurons=1000, dt=0.1, smooth=True, save_fname=None):
    """
    Function specifically for comparing the firing rate between a NMM that produces r as output and a LIF network
     that needs some formatting and filtering to get the output into the proper firing rate shape
    :param fname1: name of NMM output file
    :param fname2: name of LIF network output fil
    :param idx: idx of population
    :param n_neurons: number of neurons
    :param dt: time step
    :param smooth: bool, applying smoothing
    :param save_fname: file name to save this image to, if None, it won't be saved
    """

    with h5py.File(fname1 + '.hdf5', 'r') as h5file:
        r1 = np.array(h5file['r'])
        t = np.array(h5file['t'])
    with h5py.File(fname2 + '.hdf5', 'r') as h5file:
        r2 = np.array(h5file['r'])

    r_compare_2 = r2[idx]*(1/dt)*(1000/n_neurons)
    r_compare_1 = r1[idx] * 1000
    if smooth:
        r_compare_2 = gaussian_filter1d(r_compare_2, sigma=10)
    compare_solution(r_compare_1, r_compare_2, x=t, save_fname=save_fname, titles=[fname1, fname2])



def sigmoid(x, x0, r, amp):
    """
    Parametrized sigmoid function.

    .. math::
        y = \\frac{amp}{1+e^{-r(x-x_0)}}

    Parameters
    ----------
    x : np.ndarray of float
        (N_x) X-values the function is evaluated in.
    x0 : float
        Horizontal shift along the abscissa.
    r : float
        Slope parameter (steepness).
    amp : float
        Maximum value the sigmoid converges to.

    Returns
    -------
    y : np.ndarray of float
        (N_x) Function value at argument x.
    """
    y = amp / (1 + np.exp(-r * (x - x0)))
    return y


def DI_wave_test_function(t, intensity, t0=5, dt=1.4, width=0.25):
    """
    Determines cortical DI waves from TMS

    Parameters
    ----------
    t: ndarray of float [n_t]
        Time axis in ms
    intensity: float
        Stimulator intensity w.r.t resting motor threshold (typical range: [0 ... 2])
    t0: float
        offset time
    dt: float
        Spacing of waves in ms
    width: float
        Width of waves

    Returns
    -------
    y: ndarray of float [n_t]
        DI waves
    """

    waves = ["D", "I1", "I2", "I3", "I4"]

    x0 = dict()
    x0["D"] = 1.6952640144480995
    x0["I1"] = 1.314432218728424
    x0["I2"] = 1.4421623825084195
    x0["I3"] = 1.31643163560532
    x0["I4"] = 1.747079479469914

    amp = dict()
    amp["D"] = 12.83042571812661 / 35.46534715796085
    amp["I1"] = 35.46534715796085 / 35.46534715796085
    amp["I2"] = 26.15109003222628 / 35.46534715796085
    amp["I3"] = 15.491215097559184 / 35.46534715796085
    amp["I4"] = 10.461195366965226 / 35.46534715796085

    r = dict()
    r["D"] = 13.945868670402973
    r["I1"] = 8.707029476168504
    r["I2"] = 7.02266347578131
    r["I3"] = 16.74855628350182
    r["I4"] = 17.85806255278076

    y = np.zeros(len(t))

    for i, w in enumerate(waves):
        y_ = np.exp(-(t - t0 - i * dt) ** 2 / (2 * width ** 2))
        y_ = y_ / np.max(y_)
        y_ = y_ * sigmoid(intensity, amp=amp[w], r=r[w], x0=x0[w])
        y = y + y_

    return y

def cross_correlation_align(x1, x2, plot=False, mode='default', max_shift=int(1e12)):
    """
    Function that aligns two signals and computes the nrmse and difference between the two signals after alignment.
    Alignment is done by calculating the maximal cross correlation and then moving the second signal 2 times in the
    direction of the peak of the cross correlation to fit it on the location of the first signal, assuming the maximum
    cross correlation is in the middle between them
    :param x1: np.ndarray
                signal 1
    :param x2: np.ndarray
                signal 2
    :return:
    """
    correlation = correlate(x1, x2, mode='full')
    lag_length_1 = int(len(x2) / 2)  # cut in half, since correlation is double the size of x1 and x2
    lag_length_2 = int(len(x1) / 2)  # so lag lengths are halved to account for idx doubling
    corr_idx = int(np.argmax(correlation) / 2)  # same for correlation idx, mapping from 2*len(x1) to len(x1)
    lags = np.arange(-lag_length_1 + 1, lag_length_2)
    best_lag = lags[corr_idx] * 2  # shift two times in direction of best fit, fit is probably middle way to x1
    if abs(best_lag) > max_shift: # check if lag is within the boundary
        best_lag = max_shift * np.sign(best_lag)
    if best_lag > 0:
        aligned_signal = np.pad(x2, (best_lag, 0), mode='constant')
    elif best_lag < 0:
        aligned_signal = x2[abs(best_lag):]
        aligned_signal = np.pad(aligned_signal, (0, abs(best_lag)), mode='constant')
    else:
        aligned_signal = x2  # Already aligned

    aligned_signal = aligned_signal[:x1.shape[0]]
    if plot:
        plt.plot(x1)
        plt.plot(x2)
        plt.plot(aligned_signal)
        plt.legend(['x1', 'x2', 'aligned_x2'])
        plt.show()
    difference = np.abs(x1 - aligned_signal)
    if not mode == 'non-zero':
        error = nrmse(x1, aligned_signal)
    else:
        abs_signal = np.abs(aligned_signal)
        non_zero_mask = np.where(abs_signal > 1e-3)
        non_zero_aligned = aligned_signal[non_zero_mask]
        non_zero_x1 = x1[non_zero_mask]
        error = nrmse(non_zero_x1, non_zero_aligned)

    return error, difference, aligned_signal

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.9)
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fps, order=5):
    b, a = butter_highpass(cutoff, fps, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def t_format(time):
    """
    Function that formats a float value to a nicely readable time format
    :param time: float
        float value that represents time in seconds
    :return: t, unit: float, string
        tuple of a formated time in seconds, minutes, hours or days and the unit it is in as a string
    """
    if time < 60:
        return time, "s"
    else:
        t_min = time / 60
        if t_min < 60:
            return t_min, "min"
        else:
            t_h = t_min / 60
            if t_h < 24:
                return t_h, "h"
            else:
                t_d = t_h / 24
                return t_d, "d"


def harm_mean(x1, x2):
    """
    Compute the harmonic mean
    :param x1: np.ndaarray, array 1
    :param x2: np.ndaarray, array 2
    :return: np.ndarray, harmonic mean of x1 and x2
    """
    return 2/((1/x1)+(1/x2))

def detrend(x, y, find_peaks_args=dict(threshold=0.05, distance=1), plot=False, start_from_first_peak=False):
    """
    Function that does detrending by finding the peaks of the negative signal and
    interpolating a line for the lower bound of the signal over time. This lower bound
    is then subtracted from the original signal to detrend it. If peaks are not found, this returns the
    original signal
    :param x: np.ndarray, gird (i.e. time) of the original signal
    :param y: np.ndarray, original signal
    :return: y_detrend, np.ndarray, same shape as y: detrended version of the original signal
    """
    find_peaks_args['x'] = -y
    neg_peaks_idxs = find_peaks(**find_peaks_args)[0]
    if len(neg_peaks_idxs)>0:
        y_low = np.interp(x, x[neg_peaks_idxs], y[neg_peaks_idxs])
        if start_from_first_peak:
            start_idx = neg_peaks_idxs[0]
        else:
            start_idx = 0
        y_detrend = y.copy()
        y_detrend[start_idx:] = y[start_idx:] - y_low[start_idx:]
    else:
        y_detrend = y
    if plot:
        plt.plot(x, y)
        plt.plot(x, y_detrend)
        plt.scatter(x[neg_peaks_idxs], y[neg_peaks_idxs], color='red', marker='x')
        plt.legend(['original', 'detrended'])
        plt.show()
    return y_detrend

def plot_DI_wave_data(data, rmt_names, titles, yaxis, alphas, filter=True, do_detrend=True, sample_frequency=1e4,
                      n_channels=3, main_title='DI-wave data', emg_peak_height=200, custom_DI_wave_ylims=None,
                      switch_channel_order=False, highpass=False, hp_cutoff=0.3,
                      find_peaks_args=dict(threshold=0.05, distance=1)):
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(main_title, fontsize=16)
    n_rmts = len(rmt_names)
    if isinstance(sample_frequency, (float, int)):
        sample_frequency = [sample_frequency]*n_rmts

    n_plots = n_rmts*n_channels
    for k in range(n_rmts):
        rmt_k_data = data[rmt_names[k]][0][0][9].T
        if n_channels == 2 and switch_channel_order:
            # Manual reasigning of channels to have EMG on first and then Epidural
            rmt_k_data_new = np.zeros_like(rmt_k_data)
            rmt_k_data_new[:, 0, :] = rmt_k_data[:, 1, :]
            rmt_k_data_new[:, 1, :] = rmt_k_data[:, 0, :]
            rmt_k_data = rmt_k_data_new
        # channel 1 EMG, channel 2 & 3: EPIDURAL
        t_ch1 = np.linspace(0, rmt_k_data.shape[2]/sample_frequency[k], rmt_k_data.shape[2]) * 1e3
        mean_ch2 = rmt_k_data[:, 1, :].mean(axis=0)
        tms_peak_idx, tms_peaks = find_peaks(mean_ch2, height=emg_peak_height)
        if len(tms_peak_idx) == 0:
            tms_peak_idx, tms_peaks = find_peaks(-mean_ch2, height=emg_peak_height)
        if len(tms_peak_idx) == 0:
            raise ValueError(f'No peak corresponding to TMS detected in ch2 data of {rmt_names[k]}')
        t_peak = t_ch1[tms_peak_idx[0]]
        t_ch1 -= t_peak
        dt = np.diff(t_ch1)[0]
        t_2ms_idx = int(tms_peak_idx + (2 / dt))
        t_12ms_idx = int(tms_peak_idx + (12 / dt))
        t_5ms_idx = int(tms_peak_idx + (5 / dt))
        for j in range(n_channels):
            ax = fig.add_subplot(n_rmts, n_channels , (n_channels*k)+j+1)
            for i in range(rmt_k_data.shape[0]):
                if j > 0 and do_detrend:
                    # TODO: eventually detrend them by the mean, to have a coherent plot here
                    data_i = detrend(t_ch1, rmt_k_data[i, j], find_peaks_args=find_peaks_args)
                elif highpass and j > 0:
                    data_i = butter_highpass_filter(rmt_k_data[i, j], cutoff=hp_cutoff, fps=int(1 / dt))
                    data_i -= data_i[t_5ms_idx:t_12ms_idx].mean()
                else:
                    data_i = rmt_k_data[i, j]
                plt.plot(t_ch1, data_i, c='k', alpha=alphas[j])
            ax_mean = rmt_k_data[:, j].mean(axis=0)
            if j > 0:
                if do_detrend:
                    ax_mean = detrend(t_ch1, ax_mean, find_peaks_args=find_peaks_args)
                elif highpass:
                    ax_mean = butter_highpass_filter(ax_mean, cutoff=hp_cutoff, fps=int(1 / dt))
                    ax_mean -= ax_mean[t_5ms_idx:t_12ms_idx].mean()
                # ax_mean_filtered = butter_highpass_filter(ax_mean, cutoff=0.1, fps=int(1 / dt))
                if filter:
                    ax_mean = gaussian_filter1d(ax_mean, sigma=1)
                ax.plot(t_ch1, ax_mean, c='blue')
                ax.set_xticks(np.array([0, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10]))
                ax.set_xticklabels(['0', '2', ' ', '3', ' ', '4', ' ', '5', ' ', '6', ' ','7', ' ', '8', ' ', '9', ' ','10'])
            else:
                ax.plot(t_ch1, ax_mean, c='blue')
                ax.set_xticks(np.array([-2, 0, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])*10)
                ax.set_xticklabels(['-20', '0', '20', ' ', '30', ' ', '40', ' ', '50', ' ', '60', ' '])
            # ax.tick_params(axis='x', which='major', labelsize=7)
            if j < 1:
                ax.set_ylim(ax_mean[t_12ms_idx:].min()*1.5, ax_mean[t_12ms_idx:].max()*1.5)
                ax.set_ylabel(yaxis[k], fontsize=12)
            else:
                if do_detrend:

                    max_DI_wave = ax_mean[t_2ms_idx:t_12ms_idx].max()
                    ax.set_ylim(-0.5, max_DI_wave*1.2)
                elif highpass:
                    max_DI_wave = ax_mean[t_5ms_idx:t_12ms_idx].max()
                    min_DI_wave = ax_mean[t_5ms_idx:t_12ms_idx].min()
                    if min_DI_wave < 0:
                        ax.set_ylim(1.3*min_DI_wave, max_DI_wave * 1.3)
                    else:
                        ax.set_ylim(0.7* min_DI_wave, max_DI_wave * 1.3)
                else:
                    # ax.set_ylim(-0.2, 1)
                    ax.set_ylim(-5, 8)
                if custom_DI_wave_ylims != None:
                    ax.set_ylim(custom_DI_wave_ylims)
                ax.set_xlim(-1, 12)
                ax.vlines([5], -10, 100, color='k')
            if k < 1:
                ax.set_title(titles[j])
            if k > 1:
                ax.set_xlabel('time (ms)')
            ax.grid()

    plt.show()

def get_di_wave_data(data, rmt_names, epidural_channel_idxs, rmt_values, sample_frequency=1e4, n_channels=3, emg_peak_height=200,
                     switch_channel_order=False, find_peaks_args=dict(threshold=0.05, distance=1), meta_data=None,
                     filter_sigma=1, filter=False, do_detrend=False):

    # year + subject_ID + time - threshold -  channel -
    #  [data_DI-waves, data_DI-waves_short, threshold_type, threshold, orientation, data_name]

    n_rmts = len(rmt_names)
    if isinstance(sample_frequency, (float, int)):
        sample_frequency = [sample_frequency] * n_rmts

    if isinstance(meta_data, dict):
        di_wave_data = meta_data
        if not 'name' in meta_data.keys():
            di_wave_data['name'] = 'di_wave_data'
    else:
        di_wave_data = dict(name='di_wave_data')


    for k in range(n_rmts):
        rmt_k_data = data[rmt_names[k]][0][0][9]
        t_ch1 = np.linspace(0, rmt_k_data.shape[0] / sample_frequency[k], rmt_k_data.shape[0]) * 1e3
        # find emg channel and detect large spike related to TMS
        if epidural_channel_idxs[0] == 0:
            mean_ch_emg = rmt_k_data[:, 1, :].mean(axis=1)
        else:
            mean_ch_emg = rmt_k_data[:, 0, :].mean(axis=1)
        tms_peak_idx, tms_peaks = find_peaks(mean_ch_emg, height=emg_peak_height)
        if len(tms_peak_idx) == 0:
            tms_peak_idx, tms_peaks = find_peaks(-mean_ch_emg, height=emg_peak_height)
        if len(tms_peak_idx) == 0:
            plt.close()
            plt.plot(mean_ch_emg)
            plt.title(f'emg peak height of +/- {emg_peak_height}mV not detected')
            plt.show()
            raise ValueError(f'No peak corresponding to TMS detected in ch2 data of {rmt_names[k]}')

        t_peak = t_ch1[tms_peak_idx[0]]
        if isinstance(tms_peak_idx, (list, np.ndarray)) and len(tms_peak_idx) > 1:
            plt.plot(mean_ch_emg)
            plt.title(f'multiple emg peaks of +/- {emg_peak_height}mV detected, choosing #1')
            plt.show()
            tms_peak_idx = tms_peak_idx[0]
        t_ch1 -= t_peak
        dt = np.diff(t_ch1)[0]
        try:
            t_2ms_idx = int(tms_peak_idx + 2 / dt)
        except:
            print(tms_peak_idx)
        t_12ms_idx = int(tms_peak_idx + 12 / dt)
        t_window = t_ch1[t_2ms_idx: t_12ms_idx]

        # search for orientation in rmt_names
        if 'LM' in rmt_values[k]:
            orientation = 'LM'
        elif 'PA' in rmt_values[k]:
            orientation = 'PA'
        else:
            raise ValueError(f'Orientation not found in rmt_names: {rmt_values[k]}')
        # search for threshold type in rmt_names
        if 'RMT' in rmt_values[k]:
            threshold_type = 'RMT'
        elif 'MSO' in rmt_values[k]:
            threshold_type = 'MSO'
        else:
            raise ValueError(f'Threshold type not found in rmt_names: {rmt_values[k]}')

        # crazy re-magic to extract the string of the percentage
        RMT_digit = [int(match.group(1)) for s in rmt_values[k].split(' ') if (match := re.search(r'(\d+)%', s))]

        di_wave_data[f'{rmt_values[k]}'] = dict(time=t_ch1, time_short=t_window, RMT_digit=RMT_digit,
                                                threshold_type=threshold_type, orientation=orientation)

        for i_idx, idx in enumerate(epidural_channel_idxs):
            signals_full = rmt_k_data[:, idx, :]
            ax_mean = rmt_k_data[:, idx, :].mean(axis=1)
            # plt.plot(t_ch1, ax_mean)
            # plt.ylim(-10, 10)
            # plt.xlim(-2, 12)
            # plt.show()
            # plt.plot(t_ch1, ax_mean)
            # plt.show()
            ax_mean = ax_mean[t_2ms_idx: t_12ms_idx]
            if do_detrend:
                ax_mean = detrend(t_window, ax_mean, find_peaks_args=find_peaks_args)
            # ax_mean_filtered = butter_highpass_filter(ax_mean, cutoff=0.1, fps=int(1 / dt))
            if filter:
                ax_mean = gaussian_filter1d(ax_mean, sigma=filter_sigma)

            if 'rmt_values' in di_wave_data:
                rmt_val = di_wave_data["rmt_values"][k]
            else:
                rmt_val = rmt_names[k]
            channel_name = di_wave_data['channel_names'][idx]
            di_wave_data[rmt_values[k]][channel_name] = dict(signal_full=signals_full, signal_short=ax_mean,
                                                             data_name=rmt_names[k])
            # di_wave_data[f'{rmt_values[k]}'][di_wave_data_name_temp]['mean_window'] = ax_mean
            di_wave_full_name = f'{di_wave_data["name"]}--{rmt_val}--{channel_name}'
            # plt.plot(t_window, ax_mean)
            # plt.title(di_wave_full_name)
            # plt.show()
    return di_wave_data

def delay_signal(signal, delay, dt):
    """
    Function that delays a signal on a grid of dt spaced time points by a specific time 'delay'
    :param signal: np.ndarray, signal that wil be delayed
    :param delay:  int or float, time by which the signal will be delayed
    :param dt: float, time step of the signal, used to find the index, by which the signal is delayed
    :return: delayed_signal: np.ndarray, signal that is delayed by the specified delay value
    """
    delay_idx_value = int(delay/dt)
    delayed_signal = shift(signal, delay_idx_value, cval=0)
    return delayed_signal


def get_peak_values(x, y, find_peak_args={}, plot=False):
    """
    Function that extracts 4 types of peak values from a timeseries signal:
    t_delta_peaks: the distance between each peak in time
    amp_delta_peaks: the change in amplitude between peaks
    peak_1_time: the time delay until the first peak is observed
    peak_max_amp: the maximum amplitude found in the peaks
    area: the area under the curve of the signal

    param: x, np.ndarray: time or equivalent grid of the signal
    param: y, np.ndarray: signal
    returns:
    peak_values: dictionary of all 4 output quantities
    """
    peaks = find_peaks(y, **find_peak_args)[0]
    peak_ys = y[peaks]
    peak_xs = x[peaks]

    t_delta_peaks = np.diff(peak_xs)
    amp_delta_peaks = np.abs(np.diff(peak_ys))
    if len(peaks) > 0:
        peak_1_time = peak_xs[0]
        peak_max_amp = peak_ys.max()
    else:
        peak_1_time = np.nan
        peak_max_amp = np.nan
    area = np.sum(y)
    peak_values = dict(t_delta_peaks=t_delta_peaks,
                       amp_delta_peaks=amp_delta_peaks,
                       peak_1_time=peak_1_time,
                       peak_max_amp=peak_max_amp,
                       area=area)
    if plot:
        plt.plot(x, y)
        plt.scatter(peak_xs, peak_ys, c='green', marker='x')
        plt.grid()
        plt.show()
    return peak_values

def argmin_2d(array):
    """
    Function that returns the 2 idxs (row, col) of the smalles value in a 2D array
    :param array: np.ndarray, input array
    :return: idxs: tuple, 2 output idxs of min value
    """
    assert(len(array.shape) == 2, 'Please only provide 2D arrays as input for this funtion!')

    min_idx_1D = np.nanargmin(array)
    ncol = array.shape[1]
    row_idx = int(np.floor(min_idx_1D / ncol))
    col_idx = min_idx_1D % ncol
    return row_idx, col_idx