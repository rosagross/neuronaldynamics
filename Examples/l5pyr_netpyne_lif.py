
"""
L5 pyramidal cell population (motor cortex) using LIF neurons in NetPyNE (IntFire1).

- Population: L5Pyr (excitatory), N cells
- Neuron model: NEURON's IntFire1 (leaky integrate-and-fire artificial cell)
- Inputs: background Poisson (NetStim) -> IntFire1 events
- Optional: sparse recurrent excitation within the population (event-based)
- Outputs: raster, firing rate, spike stats

Run:
    python l5pyr_netpyne_lif.py

Requirements:
    pip install netpyne
    (and NEURON with Python support)
"""
from netpyne import specs, sim

# -----------------------
# Parameters (edit here)
# -----------------------
N = 200                     # number of L5 pyramidal cells
duration = 2000             # ms
dt = 0.1                    # ms


# Background Poisson input (NetStim)
bkg_rate = 100.              # Hz per target
bkg_weight = 0.6            # event weight -> increment of m (unitless)

# Recurrent connectivity (event-based; no synMech required for IntFire1 targets)
recurrent_p = 0.02          # connection probability
recurrent_weight = 0.4      # event weight
recurrent_delay = 1.5       # ms

# Random seeds for reproducibility
seeds = dict(conn=1, stim=2, loc=3, geom=4)

# -----------------------
# NetParams
# -----------------------
netParams = specs.NetParams()

# IntFire1 parameters (classic LIF-like artificial cell)
# Notes:
#   - 'tau' is the membrane time constant (ms) of the state variable m
#   - 'refrac' is absolute refractory period (ms)
#   - 'thresh' is the firing threshold for m (unitless; typical = 1.0)
#   - After a spike, m is reset to 0.
IF_PARAMS = dict(tau=20.0, refrac=2.0, thresh=1.0)

# Define an artificial-cell rule to pass IntFire1 parameters
# Even though artificial cells don't require sections, defining a cell rule lets us specify parameters.
cellRule = {
    'conds': {'cellType': 'L5Pyr', 'cellModel': 'IntFire1'},
    'secs': {},
    # Parameters for IntFire1 passed via point process params
    'pointps': {'if1': {'mod': 'IntFire1', **IF_PARAMS}}
}


PYR_Izhi = {'secs': {}}
PYR_Izhi['secs']['soma'] = {'geom': {}, 'pointps': {}}                        # soma params dict
PYR_Izhi['secs']['soma']['geom'] = {'diam': 10.0, 'L': 10.0, 'cm': 31.831}    # soma geometry
PYR_Izhi['secs']['soma']['pointps']['Izhi'] = {                               # soma Izhikevich properties
        'mod':'Izhi2007b',
        'C':1,
        'k':0.7,
        'vr':-60,
        'vt':-40,
        'vpeak':35,
        'a':0.03,
        'b':-2,
        'c':-50,
        'd':100,
        'celltype':1}
netParams.cellParams['L5Pyr_IF'] = PYR_Izhi
# netParams.cellParams['L5Pyr_IF'] = cellRule
# Population
netParams.popParams['L5Pyr'] = {
    'cellType': 'L5Pyr',
    'cellModel': 'IntFire1',
    'numCells': N
}

# Background Poisson input
netParams.stimSourceParams['bkg'] = {
    'type': 'NetStim',
    'rate': bkg_rate,
    'noise': 1.0,     # Poisson
    'start': 0.0
}
netParams.stimTargetParams['bkg->L5Pyr'] = {
    'source': 'bkg',
    'conds': {'cellType': 'L5Pyr'},
    # For artificial cells, weight is passed directly on the NetCon
    'weight': bkg_weight,
    # Uniform delay between given bounds
    'delay': 5,
    'synMech': 'exc'
}

netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 5.0, 'e': 0}

# feedback
if recurrent_p > 0 and recurrent_weight > 0:
    netParams.connParams['L5Pyr->L5Pyr'] = {
        'preConds': {'pop': 'L5Pyr'},
        'postConds': {'pop': 'L5Pyr'},
        'probability': recurrent_p,
        'weight': recurrent_weight,
        'delay': recurrent_delay,
        'synMech': 'exc',
        'allowSelfConns': False
    }

# -----------------------
# SimConfig
# -----------------------
simConfig = specs.SimConfig()
simConfig.duration = duration
simConfig.dt = dt
simConfig.seeds = seeds
simConfig.verbose = False
simConfig.printPopAvgRates = True

# Recordings
simConfig.recordCells = []  # all cells
simConfig.recordTraces = {}  # no Vm for artificial cells
simConfig.recordStep = dt
simConfig.recordDipole = False

# Analysis & plots
simConfig.analysis = {
    'plotRaster': {
        'orderBy': 'gid',
        'syncLines': False,
        'showFig': True,
        'saveFig': False,
        'labels': {'L5Pyr': 'L5 pyramidal (IntFire1)'},
        'figSize': (10, 4)
    },
    'plotSpikeHist': {
        'overlay': True,
        'binSize': 5,
        'showFig': True,
        'saveFig': False
    },
    'plot2Dnet': {
        'showFig': False,  # no positions by default; set to True if you add geometry
        'saveFig': False
    }
}

# -----------------------
# Build & run
# -----------------------
if __name__ == '__main__':
    sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)
