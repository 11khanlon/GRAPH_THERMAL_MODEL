"""
1st file
Stores variables in one spot so they can easily be modified
Configuration file for the Graph Theory DED thermal model

"""


#----MATERIAL PROPERTIES------

MATERIAL = {
    "name": "SteelA36",
    "density": 7.85e3,                  # rho, kg/m^3
    "ambient_temperature": 25,    #  ambient temperature,C
    "liquidus_temperature": 1500      # liquidus temperature, C
}

#--------- WAAM PARAMETERS ---------------
WAAM = {
        
    #ARC Heat input 
    "thermal_efficiency": 0.8,   # dimensionless
    "current": 129.0,             # A
    "voltage": 15.6,              # V

    # Robot travel speed
    "scan_speed": 0.762 / 60,     # m/s

    #Wire feed speed
    "WFS": 3.3/60,    #m/s

    # Deposition geometry
    "layer_height": 2.5e-3,       # m 

    # Torch visualization
    "torch_contact_tip_distance": 15e-3, #m
    "standoff_distance": 12.7e-3,  # m
    "head_height": 3.0e-3,          # m
    "head_top_radius": 1.0e-3,      # m
    "head_tip_radius": 0.8e-3,      # m

}

#--------LASER PARAMETERS---------

LASER = {

    "power": 415,            # Watts
    "scan_speed": 8.5e-3,      # m/s
    "layer_height": 0.1806e-3,   # meters
    "hatch_width": 3e-3,    # meters
    "meltpool_temperature": 2200,  # Goldak meltpool temperature, must be above liquidus and fully molten temperature
    "goldak_C": 0.171, 

    #visualization / process geometry
    "standoff_distance": 4E-3, 
    "head_height": 3.0e-3,         # m, visual geometry
    "head_top_radius": 1.0e-3,     # m
    "head_tip_radius": 0.8e-3,     # m

    "beam_radius": 0.25e-3         # m

}


#---- BUILD GEOMETRY ----

BUILD = {
    "stl_file": r"C:\\Users\\Kayleigh\\GraphThermalModel\\STL\\SingleTrak.stl",  #insertfile 

    "substrate_length": 152.4e-3,    #m
    "substrate_width": 50.8e-3,      #m
    "substrate_height": 12.7e-3,     #m

    "substrate_blocks_x": 1,
    "substrate_blocks_y": 1,
    "substrate_blocks_z": 1, 

    "blocks_per_layer": 20
    
}

#----------- BLOCK DISCRETIZATION ---------------

BLOCK = {

    # paper uses 5 blocks per hatch
    "width": 4.194e-3,    #m 
    "height": 2.501e-3,  #m 

    "surface_band_thickness": 0.1806e-3 / 2,
    "surface_thickness": 0.1806e-3 / 2,
     "time_per_block": 1
}


#------- GRAPH PARAMETERS --------

GRAPH = {
    "epsilon": 5e-3,  # neighborhood radius ε, m
    "gain": 1.3E4,     # gain factor g, m^-2
    "node_density": 0.01  # node density, nodes/mm^3
}


#----------------------------
SENSORS = {

    #Thermocouple1
    "TC_0": {
        "x": 0,
        "y": 50.8e-3,
        "z": -0.2e-3,
        "region": "substrate"
    },

    "TC_1": {
            "x": 0,
            "y": 0,
            "z": -0.2e-3,
            "region": "substrate"
        },

    "TC_2": {
                "x": 152.4e-3,
                "y": 0,
                "z": -0.2e-3,
                "region": "substrate"
            },

    "TC_3": {
                "x": 152.4e-3,
                "y": 50.8e-3,
                "z": -0.2e-3,
                "region": "substrate"
            }

}

#-------- CONVECTION ---------


CONVECTION = {
    "free": 5.0,      # W/m^2-K
    "forced": 50.0,   # W/m^2-K, free*10
}


#------- DWELL TIME ---------

DWELL = {
    "dwell_time_step": 1,    #seconds
    "dwell": 30
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
