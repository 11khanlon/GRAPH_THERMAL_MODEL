"""
config.py
Stores variables in one spot so they can easily be modified

Configuration file for the Graph Theory Directed Energy Deposition thermal model

Based on:

Riensche et al.
Thermal Modeling of Directed Energy Deposition Additive Manufacturing
Using Graph Theory
Rapid Prototyping Journal (2023)
"""

import numpy as np


#----MATERIAL PROPERTIES------

MATERIAL = {

    # Ti-6Al-4V
    "density": 4430.0,              # kg/m^3

    # W/m-K
    "thermal_conductivity": 7.2,

    # J/kg-K
    "specific_heat": 560.0,

    # ambient temperature
    "ambient_temperature": 25.0,

    # liquidus temperature
    "liquidus_temperature": 1630.0
}


#--------LASER PARAMETERS---------

LASER = {

    # Watts
    "power": 415.0,

    # m/s
    "scan_speed": 8.5e-3,

    # meters
    "beam_diameter": 1.5e-3,

    # layer height
    "layer_height": 0.1806e-3,

    # hatch width
    "hatch_width": 3e-3,

    # Goldak meltpool temperature
    "meltpool_temperature": 2200.0

}

# =============================================================================
# BUILD GEOMETRY
# =============================================================================

BUILD = {

    "length": 37.2e-3,
    "width": 3.0e-3,
    "height": 11.0e-3,

    "layers": 62,

    "substrate_length": 76.2e-3,
    "substrate_width": 25.4e-3,
    "substrate_height": 6.4e-3

}

# =============================================================================
# GRAPH PARAMETERS
# =============================================================================

GRAPH = {

    # neighborhood radius ε
    "epsilon": 0.0025,

    # gain factor γ
    "gain": 1.0,

    # node density
    "node_density": 0.470,

}

# =============================================================================
# BLOCK DISCRETIZATION
# =============================================================================

BLOCK = {

    # paper uses 5 blocks per hatch

    "length": 7.84e-3,

    "width": 3.0e-3,

    "height": 0.1806e-3,

    # seconds
    "time_per_block": 0.922

}

# =============================================================================
# CONVECTION
# =============================================================================

CONVECTION = {

    # W/m^2-K

    "forced": 50.0,

    "free": 5.0,

    "clamp": 1000.0

}

# =============================================================================
# DWELL TIME
# =============================================================================

DWELL = {

    "case_A": 20.0,

    "case_B": 3.0

}

# =============================================================================
# RANDOM NODE SETTINGS
# =============================================================================

NODES = {

    "seed": 42,

    "random": True

}

np.random.seed(NODES["seed"])