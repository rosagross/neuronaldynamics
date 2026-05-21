import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.LIF import Conductance_LIF
from Model.Nykamp_Model import Nykamp_Model_1
from Utils import plot_rates
from scipy.stats import gamma, norm, lognorm

def step(t, t0=0, t1=80):
    res = 0.7e1*np.ones_like(t)
    res[t < t0] = 0
    res[t > t1] = 0
    return res
sols = []

########################################################################################################################
# GAMMA DISTRIBUTION
########################################################################################################################

nyk_gamma = Nykamp_Model_1(parameters={'input_function': step}, name='nykamp_gamma')
nyk_gamma.simulate()
# nyk_gamma.plot()
sols.append(nyk_gamma.r[0])
nyk_gamma.clean()

########################################################################################################################
# NORMAL DISTRIBUTION
########################################################################################################################
mu_vals = np.array([0.008, 0.008])
sigma_vals = np.array([0.004, 0.004])
normal_params = {'synapse_pdf_type': 'normal', 'synapse_pdf_params': np.array([[mu_vals], [sigma_vals]]),
                 'input_function': step}

nyk_gamma = Nykamp_Model_1(parameters=normal_params, name='nykamp_normal')
nyk_gamma.simulate()
# nyk_gamma.plot()
sols.append(nyk_gamma.r[0])
nyk_gamma.clean()

########################################################################################################################
# LOG-NORMAL DISTRIBUTION
########################################################################################################################
logmu_vals = np.array([0.008, 0.008])
logsigma_vals = np.array([0.5, 0.5])
lognormal_params = {'synapse_pdf_type': 'log-normal', 'synapse_pdf_params': np.array([[logmu_vals], [logsigma_vals]]),
                 'input_function': step}

nyk_gamma = Nykamp_Model_1(parameters=lognormal_params, name='nykamp_lognormal')
nyk_gamma.simulate()
# nyk_gamma.plot()
sols.append(nyk_gamma.r[0])
nyk_gamma.clean()


a = 0.5
b = 0.008
# Sample data
gamma_samples = gamma.rvs(a=a**(-2), scale=a**2*b, size=1000)
normal_samples = norm.rvs(loc=mu_vals[0], scale=sigma_vals[0], size=1000)
lognormal_samples = lognorm.rvs(s=logsigma_vals[0], scale=logmu_vals[0], size=1000)

# Plot histograms
plt.figure(figsize=(8, 5))
plt.hist(gamma_samples, bins=30, alpha=0.5, label="Gamma Distribution", density=True)
plt.hist(normal_samples, bins=30, alpha=0.5, label="Normal Distribution", density=True)
plt.hist(lognormal_samples, bins=30, alpha=0.5, label="Log-Normal Distribution", density=True)
plt.legend()
plt.xlabel("Synaptic Conductance Jump")
plt.ylabel("Density")
plt.legend(['Gamma', 'Normal', 'Log-Normal'])
plt.show()

solutions = np.array(sols)
t = np.arange(0, nyk_gamma.T, nyk_gamma.dt)
plot_rates(solutions, t, titles=['Gamma', 'Normal', 'Log-Normal'])