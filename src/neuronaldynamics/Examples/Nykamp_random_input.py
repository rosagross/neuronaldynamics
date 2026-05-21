import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.Nykamp_Model import Nykamp_Model_1
from Utils import DI_wave_test_function, nrmse
matplotlib.use('TkAgg')

dt = 0.01
T = 100
t = np.arange(0, T, dt)
dv = 0.1
random_signal_exc = np.random.rand(t.shape[0])
random_signal_exc = 7*(random_signal_exc)

random_signal_inh = -np.random.rand(t.shape[0])
random_signal_inh = 1*(random_signal_inh)
#TODO: exc in to exc input idx, inh o inh ...


pars = {'dt': dt, 'T': T, 'input_function': random_signal_exc, 'multiple_inputs':True, 'input_function_idx':[[0, 0],
                                                                                                             [0, 1]]}
nyk1D = Nykamp_Model_1(parameters=pars, name='nyk_random')

nyk1D.simulate()
rhos = nyk1D.rho[0]
vs = nyk1D.v
rhos_end = rhos[:, -1000:]
std_end = np.std(rhos_end, axis=0)
mean_end = np.mean(rhos_end)
mean_std = np.mean(std_end)
print(f'mean std {mean_std}')
print(f'mean {mean_end}')
for i in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 980]:
    plt.plot(vs, rhos_end[:, i])
plt.show()
nyk1D.plot(heat_map=True, plot_input=False)
nyk1D.save_log()
nyk1D.clean()





