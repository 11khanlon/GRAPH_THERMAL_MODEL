"""
1st file
Stores variables in one spot so they can easily be modified
Configuration file for the Graph Theory DED thermal model

"""

import numpy as np



#----MATERIAL PROPERTIES------

MATERIAL = {

    #Ti64
    "density": 4430,              # rho, kg/m^3
    "ambient_temperature": 298.15,    #  ambient temperature, K
    "temperature_dependent":  True,
    "liquidus_temperature": 1630    # liquidus temperature, K

}


#--------LASER PARAMETERS---------

LASER = {

    "power": 415,            # Watts
    "scan_speed": 8.5e-3,      # m/s
    "beam_diameter": 1.5e-3,  # meters
    "layer_height": 0.1806e-3,   # meters
    "hatch_width": 3e-3,    # meters
    "meltpool_temperature": 2200  # Goldak meltpool temperature, must be above liquidus and fully molten temperature

}


#---- BUILD GEOMETRY ----

BUILD = {
    "stl_file": r"C://Users//Kayleigh//GRAPHTHERMALMODEL//STL//WallAssembly.stl",  #insertfile 

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

    "epsilon": 4.75e-3,  # neighborhood radius ε, m
    "gain": 1.5,     # gain factor g
    "node_density": 0.4709   # node density, nodes/mm^3

}


#----------- BLOCK DISCRETIZATION ---------------

BLOCK = {

    # paper uses 5 blocks per hatch

    "length": 7.84e-3,  #m 
    "width": 3.0e-3,    #m 
    "height": 0.1806e-3,  #m 

    "time_per_block": 0.922 #seconds

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
    #seconds 
    "case_A": 10.0,
    "case_B": 3.0

}


#------- RANDOM NODE SETTINGS ----------

NODES = {

    "seed": 42,
    "random": True

}

np.random.seed(NODES["seed"])