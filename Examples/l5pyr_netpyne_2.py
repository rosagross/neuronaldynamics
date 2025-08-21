from netpyne import specs, sim
import numpy as np
from neuron import h

netParams = specs.NetParams()

# --- Define a population of IntFire1 cells (point neurons) ---
netParams.popParams['L5Pyr'] = {
    'cellModel': 'IntFire4',      # artificial cell model
    'cellType': 'E',
    'numCells': 50,
    'params': {                   # IntFire1 parameters
        'tau': 20.0,
        'refrac': 2.0
    }
}

# --- Define a background population of NetStims (one per IntFire1 cell) ---
netParams.popParams['Bkg'] = {
    'cellModel': 'NetStim',       # NEURON's built-in NetStim generator
    'numCells': 100,               # one background stim per target cell
    'params': {
        'interval': 50,         # mean interval (ms) -> 100 Hz
        'number': 1e9,            # effectively infinite
        'start': 0.0,
        'noise': 1.0              # 1.0 = full Poisson variability
    }
}

# --- Connect background stims to L5 pyramidal IntFire1 cells ---
netParams.connParams['Bkg->L5Pyr'] = {
    'preConds': {'pop': 'Bkg'},
    'postConds': {'pop': 'L5Pyr'},
    'weight': 1.0,    # increment of state variable m
    'delay': 1.0,
    'probability': 0.01
    # no synMech for artificial cells!
}
#
# # --- (optional) add sparse recurrent excitation within L5Pyr ---
# netParams.connParams['L5Pyr->L5Pyr'] = {
#     'preConds': {'pop': 'L5Pyr'},
#     'postConds': {'pop': 'L5Pyr'},
#     'probability': 0.02,
#     'weight': 0.01,
#     'delay': 1.5
# }

# This is not working!
# TODO: this seems like a tough workaround!
def add_time_varying_input(sim):
    tvec = h.Vector(np.arange(0, simConfig.duration, simConfig.dt))  # time in ms
    ivec = h.Vector(0.5 * (1 + np.sin(50 * np.pi * tvec.as_numpy())))  # example waveform

    for cell in sim.net.cells:
        if cell.tags['cellModel'] == 'IntFire4':
            ivec.play(cell.hPointp._ref_i, tvec, 1)


# Register this function


# --- Simulation config ---
simConfig = specs.SimConfig()
simConfig.duration = 400
simConfig.dt = 0.1
simConfig.verbose = True
simConfig.printPopAvgRates = True
simConfig.analysis['plotRaster'] = {'showFig': True, 'saveFig': True}
simConfig.createPyFuncs = [add_time_varying_input]
simConfig.validateNetParams = True

# --- Run ---
sim.createSimulateAnalyze(netParams, simConfig)
