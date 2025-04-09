import pyrates
import numpy as np
import matplotlib.pyplot as plt
import pygpc
import os
import h5py
import matplotlib
matplotlib.use("Qt5Agg")
def main():

    # define simulation time and input start and stop
    T = 8.0
    step_size = 5e-4
    step_size_sampling = 1e-3
    start = 1.0
    stop = 5.0

    # extrinsic input definition
    steps = int(np.round(T/step_size))
    I_ext = np.zeros((steps,))
    I_ext[int(start/step_size):int(stop/step_size)] = 3.0

    # equations for a QIF population
    qif = ['d/dt * r = (delta/(pi*tau) + 2.*r*v) /tau',
           'd/dt * v = (v^2 + eta + I_ext/g_mem + J*s*tau - (pi*r*tau)^2) /tau']

    # neuron variables
    variables = {'delta': 1.0,
                 'tau': 0.13,
                 'g_mem': 1.44,
                 'eta': -5.0,
                 'J': 15.0,
                 'r': 'output',
                 'v': 'variable',
                 'I_ext': 'input',
                 's': 'input'}

    # operator setup
    qif_op = pyrates.OperatorTemplate(name='qif_op', path=None, equations=qif, variables=variables)
    # set up the node template
    pop_exc = pyrates.NodeTemplate(name='L5P', path=None, operators=[qif_op])

    # set up the circuit template
    circuit = pyrates.CircuitTemplate(name='PC', path=None, nodes={'L5P': pop_exc},
                                      edges=[('L5P/qif_op/r', 'L5P/qif_op/s', None, {'weight': 1.0})])

    # perform a numerical simulation with input delivered to u and recordings of v and r
    results = circuit.run(simulation_time=T,
                          step_size=step_size,
                          inputs={"L5P/qif_op/I_ext": I_ext},
                          outputs={"r": "L5P/qif_op/r",
                                   "v": "L5P/qif_op/v"},
                          sampling_step_size=step_size_sampling,
                          in_place=False,
                          clear=True)

    fig, axes = plt.subplots(nrows=2, figsize=(12, 6))

    # plot the firing rate in one axis
    axes[0].plot(results['r'])
    axes[0].set_ylabel('r in Hz')
    axes[0].grid()

    # plot the membrane potential in the other axis
    axes[1].plot(results['v'])
    axes[1].set_xlabel('t in s')
    axes[1].set_ylabel('v in mV')
    axes[1].grid()
    axes[0].set_title('single exc population L5P cells')
    plt.tight_layout()
    plt.show()
    plt.close()


    # time axis
    # for time in s add factor 1e-3
    factor = 1e-3
    t = factor * np.linspace(0, 99.81, 500)
    dt = np.diff(t)[0]
    T = t[-1] + dt

    # scaling factor for current (gpc was done in normalized current space)
    i_scale = 10  # 5.148136e-09

    # read gpc session
    fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
    session = pygpc.read_session(fname=fn_session)

    with h5py.File(os.path.splitext(fn_session)[0] + ".hdf5", "r") as f:
        coeffs = f["coeffs"][:]

    # create grid object to transform from real to normalized coordinates [-1, 1]
    theta = 0               # angle of e-field [0, 180]°
    gradient = 0            # relative gradient of e-field [-20, 20] %/mm
    intensity = 250         # intensity of e-field [100, 400] V/m
    fraction_nmda = 0.5     # fraction of nmda synapses [0.25, 0.75]
    fraction_gaba_a = 0.95  # fraction of nmda synapses [0.9, 1.0]
    fraction_ex = 0.5       # fraction of exc/ihn synapses [0.2, 0.8]

    coords = np.array([[theta, gradient, intensity, fraction_nmda, fraction_gaba_a, fraction_ex]])

    grid = pygpc.RandomGrid(parameters_random=session.parameters_random, coords=coords)

    # use gpc approximation to compute current
    current = session.gpc[0].get_approximation(coeffs, grid.coords_norm) * i_scale
    current = current.flatten()

    # plot current
    plt.plot(t*factor, current)
    plt.show()

    def sin_current(t):
        v0 = 0.1
        f = 100
        return v0*(1+np.sin(2*np.pi*f*t))

    current = sin_current(t)
    # current = np.zeros_like(t)


    # for pickle, beware __randomstate_ctor() takes from 0 to 1 positional arguments but 2 were given

    results = circuit.run(simulation_time=T,
                          step_size=dt,
                          inputs={"L5P/qif_op/I_ext": current},
                          outputs={"r": "L5P/qif_op/r",
                                   "v": "L5P/qif_op/v"},
                          sampling_step_size=dt,
                          in_place=False,
                          clear=True)

    fig, axes = plt.subplots(nrows=2, figsize=(12, 6))

    # plot the firing rate in one axis
    axes[0].plot(results['r'])
    axes[0].set_ylabel('r in Hz')
    axes[0].grid()

    # plot the membrane potential in the other axis
    axes[1].plot(results['v'])
    axes[1].set_xlabel('t in s')
    axes[1].set_ylabel('v in mV')
    axes[1].grid()
    axes[0].set_title('single exc population L5P cells')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()