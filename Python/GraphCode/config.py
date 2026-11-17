"""
Stores variables in one spot so they can easily be modified
Configuration file for the Graph Theory DED thermal model
Make sure you run the new data through synchronize_thermal_data.py first to get the correct dwell time for the pass you are analyzing
"""

from pathlib import Path
import pandas as pd


#----MATERIAL PROPERTIES------

MATERIAL = {
    "name": "SteelA36",
    "material_directory": "A36Steel_Info",
    "density": 7.85e3,                  # rho, kg/m^3
    "ambient_temperature": 25,    #  ambient temperature,C
    "liquidus_temperature": 1500      # liquidus temperature, C
}


IMAGE_PROCESSING = {
     "PRE_BUFFER_S": 1,  # seconds
     "NUMBER_OF_PASSES": 1   #this is just the number of desired passes you want to analyze
     #Make sure to rename the actual filepaths for thermal_image_processor and use the count to select desired images 
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
    "stl_file": r"C:\\Users\\khanlon6\GRAPH_THERMAL_MODEL\\STL\\SingleTrak.stl",  #insertfile 
    #"stl_file": r"C:\\Users\\Kayleigh\\GraphThermalModel\\STL\\SingleTrak.stl",  #insertfile 

    "substrate_length": 170.0e-3,    #m
    "substrate_width": 76.20e-3,      #m
    "substrate_height": 12.7e-3,     #m

    "substrate_blocks_x": 1,
    "substrate_blocks_y": 1,
    "substrate_blocks_z": 1, 

    "blocks_per_layer": 10,   #simulation blocks. 10 blocks per layer
    
    "bead_length": 140/1000,  #m
    "x_start": 15.207/1000   #m
}

#----------- BLOCK DISCRETIZATION ---------------

BLOCK = {

    # paper uses 5 blocks per hatch
    "width": 4.194e-3,    #m 
    "height": 2.501e-3,  #m 

    "surface_band_thickness": 0.5e-3 /2 ,
    "surface_thickness": 0.5e-3 / 2 ,
     "time_per_block": 1
}


#------- GRAPH PARAMETERS --------

GRAPH = {

    "seed": 42,
    "epsilon": 3e-3,  # neighborhood radius ε, m
    "gain": 1E7,     # gain factor g, m^-2
    "node_density": 0.01 # node density, nodes/mm^3
}


#----------------------------
SENSORS = {

    #Thermocouple1
    "TC_0": {
        "x": 135.80e-3,
        "y": 8.382e-3,
        "z": 0,
        "region": "substrate"
    },

    "TC_1": {
            "x": 135.80e-3,
            "y": 59.436e-3,
            "z": 0,
            "region": "substrate"
        },

    "TC_2": {
                "x": 34.20e-3,
                "y": 8.382e-3,
                "z": 0,
                "region": "substrate"
            },

    "TC_3": {
                "x": 34.20e-3,
                "y": 59.436e-3,
                "z": 0,
                "region": "substrate"
            }
}

#-------- CONVECTION ---------


CONVECTION = {
    "free": 5.0,      # W/m^2-K
    "forced": 50.0,   # W/m^2-K, free*10
}


# ------- DWELL TIME ---------

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

PASS_INFO_FILE = (
    PROJECT_DIR
    / MATERIAL["material_directory"]
    / "ThermalData"
    / "Processed"
    / "Pass_1"
    / "Pass1_Info.csv")

DWELL = {
    "dwell_time_step": 1.0,
    "dwell": None
}


def load_experimental_dwell():

    if not PASS_INFO_FILE.exists():
        raise FileNotFoundError(
            f"Experimental dwell metadata not found: {PASS_INFO_FILE}\n"
            "Run synchronize_thermal_data.py first.")

    pass_info = pd.read_csv(PASS_INFO_FILE)
    dwell_duration = float(pass_info.loc[0, "DwellDuration_s"])
    DWELL["dwell"] = dwell_duration

    print(f"Experimental dwell duration: {DWELL['dwell']:.3f} s")

    return dwell_duration



