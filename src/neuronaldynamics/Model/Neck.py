import numpy as np
import matplotlib as plt
import matplotlib
import scipy
import os
matplotlib.use('TkAgg')

# current_path = os.path.abspath(__file__)
# split_path = current_path.split('\\')[:-1]
# current_directory = '\\'.join(split_path)
current_directory = os.path.dirname(__file__)
def generate_EP(d=0.01, plot=False, Axontype=1, N=1000, dt=1e-3):
    # Load AP data based on Axontype
    if Axontype == 1:
        AP_fname = os.path.join(current_directory, 'AP.mat')
        data = scipy.io.loadmat(AP_fname)
    elif Axontype == 2:
        AP_fname = os.path.join(current_directory, 'AP2.mat')
        data = scipy.io.loadmat(AP_fname)

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
    dt_new = dt # old version with dt
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


def EP_convolve(x, t, dt=0.1, scale=1, plot=False):
    EP, t_EP, AP_out = generate_EP(d=0.1, plot=False, Axontype=1, dt=dt * 10)
    EP = -EP
    EP = EP / np.max(EP)
    EP_small = np.interp(t[t < 1.0] - 0.5, t_EP, EP)
    x_potential = scipy.signal.convolve(x, EP_small)
    x_shape = x.shape[0]
    x_potential_out = x_potential[:x_shape]
    x_potential_scaled = x_potential_out / np.max(x_potential_out) * scale
    if plot:
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(t, x)
        ax[0].set_ylabel('DI wave potential')
        ax[1].plot(t[:EP_small.shape[0]], EP_small)
        ax[1].set_ylabel('Kernel')
        ax[2].plot(t, x_potential_scaled)
        ax[2].set_ylabel('DI wave rate')
        for i in range(3):
            ax[i].set_xlabel('t (ms)')
            ax[i].set_xlim([t[0], t[-1]])
        plt.show()
    return x_potential_scaled
