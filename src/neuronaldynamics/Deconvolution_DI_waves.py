import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import signal
import scipy.spatial.distance
import skimage

def generate_EP(d=0.01, plot=False, Axontype=1, N=1000):
    # Load AP data based on Axontype
    if Axontype == 1:
        data = scipy.io.loadmat('AP.mat')
    elif Axontype == 2:
        data = scipy.io.loadmat('AP2.mat')

    TIME_VECTOR = data['TIME_VECTOR'].flatten()
    MEMBRANE_POTENTIAL = data['MEMBRANE_POTENTIAL']

    # Selecting action potential index
    idx = 20
    times = TIME_VECTOR
    padding = 1000
    dt = times[1] - times[0]
    extend_time = times[-1] + np.arange(1, padding + 1) * dt
    extend_potential = np.ones(padding) * MEMBRANE_POTENTIAL[-1, idx]

    # Extending time and voltage array for padding
    times = np.concatenate((times, extend_time))
    v = np.concatenate((MEMBRANE_POTENTIAL[:, idx], extend_potential))

    # First and second derivatives
    dv = np.gradient(v)
    ddv = np.gradient(dv)

    # Calculate extracellular potential (EP)
    EP = np.zeros((ddv.shape[0]))
    # dists = scipy.spatial.distance.cdist(times, times)
    # masks = 1/dists # needs the d offset for some reason?
    for t in range(times.shape[0]):
        mask = 1 / np.sqrt((times - times[t]) ** 2 + d ** 2)
        EP[t] = np.sum(ddv * mask)

    # Plot results if plot is True
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(5, 8))
        axes[0].plot(times, v)
        axes[0].set_ylabel("mV")
        axes[0].set_title("Action potential by AC model (V)")

        axes[1].plot(times, dv)
        axes[1].set_title("V'")

        axes[2].plot(times, ddv)
        axes[2].set_title("V''")

        axes[3].plot(times, EP)
        axes[3].set_title(f"Extracellular potential (d={d})")
        axes[3].set_xlabel("Time (ms)")

        plt.tight_layout()
        plt.show()

    # Prepare output data
    dt_new = 1e-3  # old version with dt
    times_new = np.arange(-0.5, 0.5 + dt_new, dt_new)
    # times_new = np.linspace(-0.7, 0.7, int(N))

    AP2 = v - np.min(v)
    max_AP_idx = np.argmax(AP2)
    max_AP = np.max(AP2)
    min_EP_idx = np.argmin(EP)
    min_EP = np.min(EP)

    AP2 /= max_AP
    AP2 = np.interp(times_new, times - times[max_AP_idx], AP2)
    EP2 = EP / abs(min_EP)
    EP2 = np.interp(times_new, times - times[min_EP_idx], EP2)

    return EP2, times_new, AP2

def sigmoid(x, x0, r, a):
    return a / (1 + np.exp(r * (x0 - x)))

def gen_DIwave(t, intensity):
    t0 = 5  # Offset
    T = 1.5  # Peak interval
    width = 0.25  # Wave width

    DIwave = np.zeros_like(t)

    # D, I1, I2, I3, I4 parameters
    x0 = np.array([1.36192637, 1.04127548, 1.16603639, 1.03733872, 1.45405986])
    r = np.array([18.50774852, 9.26210842, 5.91559859, 17.7805388, 425.51252596])
    a = np.array([0.34532065, 1.0, 0.80577286, 0.46054753, 0.27828232])

    # Compute DIwave using summation of Gaussian functions weighted by sigmoid
    for i in range(5):
        DIwave += np.exp(-(t - t0 - (i * T)) ** 2 / (2 * width ** 2)) * sigmoid(intensity, x0[i], r[i], a[i])

    return DIwave

def regularized_deconvolution(signal, kernel, reg_param=0.01):
    signal_shape = signal.shape[0]
    kernel_shape = kernel.shape[0]
    n_kernel_padding = signal_shape - kernel_shape
    kernel_padding = np.zeros(n_kernel_padding)
    kernel = np.hstack((kernel, kernel_padding))
    kernel_fft = np.fft.fft(kernel)
    signal_fft = np.fft.fft(signal)
    deconvolved_fft = signal_fft / (kernel_fft + reg_param)
    return np.fft.ifft(deconvolved_fft).real, 0

plot_scatter = False
# Data preparation
x = 3 / 80 * np.arange(-3, 31, 3) + 1  # AMT average
x *= 0.75  # AMT to RMT
z = np.linspace(0.55, 1.7, 1000)  # Continuous range for fitting

# Data values
data = {
    "I1": [0, 2.609, 2.958, 10.57, 16.79, 18.82, 30.33, 30.97, 37.46, 33.35, 36.22, 36.85],
    "I2": [0, 0, 0, 10.06, 9.113, 11.17, 14.89, 18.86, 22.56, 21.34, 25.60, 29.58],
    "I3": [0, 0, 0, 0, 7.659, 10.90, 16.10, 15.69, 16.69, 13.47, 16.71, 21.34],
    "I4": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11.44, 8.991],
    "D": [0, 0, 0, 0, 0, 0, 0, 0, 6.389, 9.444, 10.00, 14.17]
}

colors = ["blue", "orange", "green", "red", "purple"]

# Plot scatter & sigmoid curve fitting
if plot_scatter:
    plt.figure(figsize=(10, 8))
    for label, y in data.items():
        plt.scatter(x, y, label=label, color=colors[list(data.keys()).index(label)], alpha=0.7)
        p0 = [1.5, 10, 20]
        bounds = ([1, 0, 0], [2.5, 500, 40])

        popt = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds)
        plt.plot(z, sigmoid(z, popt[0][0], popt[0][1], popt[0][2]), color=colors[list(data.keys()).index(label)], linewidth=1.5)

    plt.legend()
    plt.xlabel("TMS intensity (RMT)")
    plt.ylabel("Amplitude (μV)")
    plt.title("Sigmoid Curve Fit")
    plt.show()

# Generate DIwaves
dt = 0.01  # ms
t = np.arange(0, 15, dt)
TMS_intensities = [0.8, 1, 1.5]
n_intensities = len(TMS_intensities)

di_waves = np.array([gen_DIwave(t, intensity) for intensity in TMS_intensities]).T

# Interpolate DIwave
times_new = np.arange(t[0], t[-1] + dt, dt)
times_new_positive = times_new[times_new > 0]

di_waves_interp = np.zeros_like(di_waves)
for i in range(n_intensities):
    di_waves_interp[:, i] = np.interp(times_new, t, di_waves[:, i].T).T

# Generate EP
d = 0.1
kernel_length = int(times_new.shape[0])
EP, t_EP, AP_out = generate_EP(d, plot=False, Axontype=1, N=kernel_length)
EP = -EP

padding = EP.shape[0]
if padding < 0:
    raise ValueError('Kernel dimension is bigger than signal dimension!')
# di_wave_padding = np.ones((padding, n_intensities)) * di_waves_interp[0, :]
di_wave_padding = 1e-50*np.ones((padding, n_intensities))
di_waves_extended = np.vstack([di_wave_padding, di_waves_interp])


# Deconvolution
rate = np.zeros_like(di_waves_interp)
for i in range(di_waves_extended.shape[1]):
    deco = regularized_deconvolution(di_waves_extended[:, i], EP, reg_param=0.1)[0][:times_new.shape[0]]
#     deco = signal.deconvolve(di_waves_extended[:, i], EP)[0][:-1]
    rate[:, i] = deco / np.max(deco)
# repeated_EP = EP.repeat(3).reshape((EP.shape[0], 3))
# p0 = {'prior': di_waves_interp}
# rate = skimage.restoration.unsupervised_wiener(di_waves_interp, repeated_EP, user_params=p0)[0]

# Plot results
intensities = ['0.8', '1.0', '1.5']
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for i in range(di_waves_interp.shape[1]):
    # norm_curve = di_waves_interp[:, i] / np.max(di_waves_interp[:, i])
    axes[0].plot(t, di_waves_interp[:, i], linewidth=2)
axes[0].set_title("DI-waves")
axes[0].set_ylabel("Potential")
axes[0].legend(intensities)

axes[1].plot(t_EP - t_EP[0], EP, 'k', linewidth=1.2)
axes[1].set_title(f"EP (d/c={d})")
axes[1].set_ylabel("Potential")

for i in range(rate.shape[1]):
    axes[2].plot(times_new, rate[:, i] / np.max(rate[:, i]), linewidth=2)
axes[2].set_title("Deconvolved DI-waves")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Firing Rate")
axes[2].legend(intensities)

for i in range(3):
    axes[i].set_xlim([t[0], t[-1]])
    axes[i].grid()


plt.tight_layout()
plt.show()


