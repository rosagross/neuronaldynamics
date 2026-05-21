from pyrates.frontend import CircuitTemplate
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')


wc = CircuitTemplate.from_yaml("model_templates.neural_mass_models.wilsoncowan.WC")

# simulation parameters
T = 125.0
dt = 5e-4
dts = 1e-2
steps = int(np.round(T/dt))
inp = np.zeros((steps,))
inp[int(25/dt):int(100/dt)] = 1.0

# perform simulation
results = wc.run(simulation_time=T,
                 step_size=dt,
                 sampling_step_size=dts,
                 outputs={'E': 'e/rate_op/r',
                          'I': 'i/rate_op/r'},
                 inputs={'e/se_op/r_ext': inp},
                 backend='default',
                 solver='euler')

inps = np.zeros(int(np.round(T/dts)))
inps[int(25/dts):int(100/dts)] = 1.0
sample_time = np.asarray(results.index)
plt.plot(results)
plt.plot(sample_time, inps)
plt.legend(results.keys() + ['input'])
plt.ylabel('result')
plt.xlabel('t in s')
plt.show()

# simulate biphasic TMS pulse
in_TMS = np.zeros_like(inps)

start = 2000
end = 6000
for i in range(start, end):
    f = 1.5
    phi_0 = f*start
    in_TMS[i] = np.sin(0.001*f*i - phi_0)
# plt.plot(sample_time, in_TMS)

results_1 = wc.run(simulation_time=T,
                 step_size=dts,
                 sampling_step_size=dts,
                 outputs={'E': 'e/rate_op/r',
                          'I': 'i/rate_op/r'},
                 inputs={'e/se_op/r_ext': in_TMS},
                 backend='default',
                 solver='euler')

#TODO: find out why this doesn't work

plt.plot(results_1)
plt.plot(sample_time, in_TMS)
plt.legend(results_1.keys() + ['input'])
plt.ylabel('result')
plt.xlabel('t in s')
plt.show()