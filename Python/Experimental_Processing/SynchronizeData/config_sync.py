# Processing settings
from pathlib import Path
import sys

#1. Obtain GraphCode Python directory
BASE_DIR = Path(__file__).resolve().parent
PYTHON_DIR = BASE_DIR.parent.parent
sys.path.insert(0, str(PYTHON_DIR))
from GraphCode.config import BUILD, MATERIAL, WAAM, IMAGE_PROCESSING



thermal_metadata = {
    "material_directory": MATERIAL["material_directory"], 
    "PRE_BUFFER_S": IMAGE_PROCESSING["PRE_BUFFER_S"],  # seconds
    "NUMBER_OF_PASSES": IMAGE_PROCESSING["NUMBER_OF_PASSES"],   #this is just the number of desired passes you want to analyze
    "SCAN_SPEED_M_S": WAAM["scan_speed"],
    "BEAD_LENGTH_M": BUILD["bead_length"],
    "x_start": BUILD["x_start"]
}




print("Thermal metadata:", thermal_metadata)