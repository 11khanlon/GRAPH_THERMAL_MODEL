# Processing settings
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent

# GraphThermalModel/Python
PYTHON_DIR = BASE_DIR.parent.parent

# GraphThermalModel/Python/GraphCode
GRAPH_CODE_DIR = PYTHON_DIR / "GraphCode"

sys.path.insert(0, str(GRAPH_CODE_DIR))

from config import BUILD, WAAM, MATERIAL



thermal_metadata = {
    "material_directory": MATERIAL["material_directory"], 
    "PRE_BUFFER_S": 1,
    "NUMBER_OF_PASSES": 2,   #this is just the number of desired passes you want to analyze
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