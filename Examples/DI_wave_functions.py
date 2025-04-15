import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")


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


def DI_wave(t, intensity, t0=5, dt=1.4, width=0.25):
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


# time axis
t = np.linspace(0, 20, 1000)

# DI waves for different intensities
y1 = DI_wave(t, intensity=1, t0=5, dt=1.4, width=0.25)
y2 = DI_wave(t, intensity=1.5, t0=5, dt=1.4, width=0.25)
y3 = DI_wave(t, intensity=2, t0=5, dt=1.4, width=0.25)


# normalize DI waves (we do not have quantitative information from the papers)
y1 /= np.max(y1)
y2 /= np.max(y2)
y3 /= np.max(y3)


# plot
fig, ax = plt.subplots(3, 1, figsize=[5, 7])
ax[0].plot(t, y1)
ax[1].plot(t, y2)
ax[2].plot(t, y3)
ax[0].legend(["intensity=1"])
ax[1].legend(["intensity=1.5"])
ax[2].legend(["intensity=2.0"])
ax[2].set_xlabel("t in ms")
ax[2].set_xlabel("DI wave amplitude (normalized)")
plt.tight_layout()
plt.show()
