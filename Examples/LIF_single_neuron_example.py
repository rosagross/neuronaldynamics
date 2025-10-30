import numpy as np
import matplotlib.pyplot as plt


def LIF_neuron(t, v_rest, tau, v_ext, v_th=-55, tau_ref=1.0):
    t_shape = t.shape[0]
    dt = np.mean(np.diff(t))
    v = np.zeros_like(t)
    ref_counter = 0
    t_steps_ref = int(tau_ref/dt)
    spike_count = 0
    t_spike = []

    for i in range(1, t_shape):

        if ref_counter > 0:
            ref_counter -= 1
            v[i] = v_rest
        else:
            dv = (1/tau)*(-(v[i-1] - v_rest) + v_ext[i-1])*dt
            v[i] = v[i-1] + dv
        if v[i] >= v_th:
            v[i] += 20
            ref_counter += t_steps_ref
            spike_count += 1
            t_spike.append(t[i])

    spike_interval = t_spike[-1] - t_spike[0]
    f = spike_count / spike_interval * 1e3  # convert to Hz from mHz

    return v, f


# units in ms, mV
T = 10
t = np.linspace(0, T, 1000)
v_ext = np.zeros_like(t)
t_mask = (t >= 2) & (t <= 8)
v_ext[t_mask] = 5e4
tau = 12
tau_ref = 1.5
v_rest = -70
v_thr = -55
v, f = LIF_neuron(t, v_rest, tau, v_ext, v_thr, tau_ref)

#######################################################
# PLOTTING                                            #
#######################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, v)
ax2 = ax.twinx()
ax.hlines(y=v_thr, xmin=t[0], xmax=t[-1])
ax.set_ylabel('v (mV)')
ax.set_xlabel('t (ms)')
ax.set_ylim([-80, -40])
ax.set_xlim([t[0], t[-1]])
ax2.plot(t, v_ext, c='green')
ax2.legend(['v_ext'])
ax2.set_ylabel('v_ext in (mV)')
ax2.set_ylim([-v_ext.max(), v_ext.max()*3])
bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
text = ax.text(0.98, 0.85, f'f = {f:.1f}Hz', fontsize=9, bbox=bbox,
        transform=ax.transAxes, horizontalalignment='right')
plt.show()
