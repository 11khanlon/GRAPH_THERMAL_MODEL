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


'''
There will be a prebuffer of 1 second 
the dwell, however, will not be hardcoded. this dwell will be stored at the time in between the arc is on and of in between layers. This is found form the PROCESSACTIVE Tab. 
When the torch turns from True to false here once layer one is complete, the dwell period starts. when the torch turns from false to true, the dwell period ends. This will be stored in a variable called DWELL_S.
I dont think i need a post dwell buffer. what what this used for again? 
for each pass, the idea is add a 1second prebuffer, then deposit material (processactive = True), then dwell for the time in between layers (DWELL_S). Create another layer, then dwell.


'''


print("Thermal metadata:", thermal_metadata)