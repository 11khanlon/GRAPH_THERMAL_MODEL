"""
1st file
Stores variables in one spot so they can easily be modified
Configuration file for the Graph Theory Directed Energy Deposition thermal model

"""

import numpy as np


#----MATERIAL PROPERTIES------

MATERIAL = {

    "density": 8190,              # rho, kg/m^3
    "thermal_conductivity": 18.0,    # k, W/m-K
    "specific_heat": 550,         # cp,  J/kg-K
    "ambient_temperature": 298.15,    #  ambient temperature, K
    "liquidus_temperature": 1609.0    # liquidus temperature, K
}


#--------LASER PARAMETERS---------

LASER = {

    "power": 1070.0,            # Watts
    "scan_speed": 12.7e-3,      # m/s
    "beam_diameter": 1.778e-3,  # meters
    "layer_height": 0.381e-3,   # meters
    "hatch_width": 1.143e-3,    # meters
    "meltpool_temperature": 2000  # Goldak meltpool temperature, must be above liquidus and fully molten temperature

}


#---- BUILD GEOMETRY ----

BUILD = {

    "length": 37.2e-3,
    "width": 3.0e-3,
    "height": 11.0e-3,

    "layers": 62,

    "substrate_length": 76.2e-3,
    "substrate_width": 25.4e-3,
    "substrate_height": 6.4e-3

}


#------- GRAPH PARAMETERS --------

GRAPH = {

    "epsilon": 0.0025,  # neighborhood radius ε
    "gain": 1.0,     # gain factor g
    "node_density": 0.470,     # node density, nodes/mm^3

}


#----------- BLOCK DISCRETIZATION ---------------

BLOCK = {

    # paper uses 5 blocks per hatch

    "length": 7.84e-3,

    "width": 3.0e-3,

    "height": 0.1806e-3,

    # seconds
    "time_per_block": 0.922

}


#-------- CONVECTION ---------


CONVECTION = {

    # W/m^2-K

    "forced": 50.0,
    "free": 5.0,
    "clamp": 1000.0   #do you need a clamp in this study 

}


#------- DWELL TIME ---------


DWELL = {

    "case_A": 10.0,
    "case_B": 3.0

}


#------- RANDOM NODE SETTINGS ----------

NODES = {

    "seed": 42,
    "random": True

}

np.random.seed(NODES["seed"])