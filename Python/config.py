"""
1st file
Stores variables in one spot so they can easily be modified
Configuration file for the Graph Theory DED thermal model

"""


#----MATERIAL PROPERTIES------

MATERIAL = {

    "name": "Ti64",
    "density": 4430,                  # rho, kg/m^3
    "ambient_temperature": 25,    #  ambient temperature,C
    "temperature_dependent":  True,
    "liquidus_temperature": 1630      # liquidus temperature, C

}


#--------LASER PARAMETERS---------

LASER = {

    "power": 415,            # Watts
    "scan_speed": 8.5e-3,      # m/s
    "beam_diameter": 1.5e-3,  # meters
    "layer_height": 0.1806e-3,   # meters
    "hatch_width": 3e-3,    # meters
    "meltpool_temperature": 2200,  # Goldak meltpool temperature, must be above liquidus and fully molten temperature
    "goldak_C": 0.171, 
    "thermal_conductivity": 6.8 ,  #W/mK
    "thermal_diffusivity": 2.7228E-6 , #m^2/s, 
    "standoff_distance": 11.4E-3 

}


#---- BUILD GEOMETRY ----

BUILD = {
    "stl_file": r"C:\\Users\\Kayleigh\\GraphThermalModel\\STL\\Wall.stl",  #insertfile 

    "substrate_length": 76.2e-3,
    "substrate_width": 25.4e-3,
    "substrate_height": 6.4e-3,

    "blocks_per_layer": 5
    
}


#------- GRAPH PARAMETERS --------

GRAPH = {

    "epsilon": 4.75e-3,  # neighborhood radius ε, m
    "gain": 1.5E6,     # gain factor g, m^-2
    "node_density": 0.4709  # node density, nodes/mm^3

}


#----------- BLOCK DISCRETIZATION ---------------

BLOCK = {

    # paper uses 5 blocks per hatch
    "length": 7.84e-3,  #m 
    "width": 3.0e-3,    #m 
    "height": 0.1806e-3,  #m 

    "surface_band_thickness": 0.1806e-3 / 2,
    "surface_thickness": 0.1806e-3 / 2, 

    "time_per_block": 0.922 #seconds

}


#-------- CONVECTION ---------


CONVECTION = {

    "free": 5.0,      # W/m^2-K
    "forced": 50.0,   # W/m^2-K, free*10
    "clamp": 1000.0   #do you need a clamp in this study 

}


#------- DWELL TIME ---------

DWELL = {
    
    "case_A": 20.0,  #seconds 
    "case_B": 3.0 ,   #seconds 
    "time_step": 1    #seconds
}


if __name__ == "__main__":

    import numpy as np

    Bi_free = CONVECTION["free"]*BLOCK["length"] / LASER["thermal_conductivity"]
    Bi_forced = CONVECTION["forced"]*BLOCK["length"] / LASER["thermal_conductivity"]

    if Bi_free < 0.1:
        print(f"\nConduction resistance is less than convection ✓ ")
        print(f" Biot number for free convection:", Bi_free)

    if Bi_forced < 0.1:
            print(f"\nConduction resistance is less than convection ✓ ")
            print(f" Biot number for forced convection:", Bi_forced)

    print(f"\n---- Needed subtrate depth -------")
    number_of_blocks = 5  #worst case scenario for heat accumulation 

    Ld = 2 * np.sqrt(LASER["thermal_diffusivity"] * BLOCK["time_per_block"])

    print(f"Needed substrate depth in mm :", number_of_blocks* Ld * 1000)
