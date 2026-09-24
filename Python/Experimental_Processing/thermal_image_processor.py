from pathlib import Path
import cv2
import numpy as np
from spatial_calibration import SpatialCalibration


# PATHS
BASE_DIR = Path(__file__).resolve().parent
THERMAL_DATA_DIR = (BASE_DIR.parent / "ThermalData")
CALIBRATION_IMAGE = (THERMAL_DATA_DIR / "Substrate_Calibration.tiff")

PASS_DIRS = {
    1: THERMAL_DATA_DIR / "Thermal_Images_Pass1",
    2: THERMAL_DATA_DIR / "Thermal_Images_Pass2",
}

OUTPUT_DIR = (THERMAL_DATA_DIR / "Processed" / "Spatial")
CALIBRATION_FILE = (OUTPUT_DIR / "spatial_calibration.npz")


# ============================================================
# OUTPUT DIRECTORIES -- Need to fix this 

for pass_number in [1, 2]:
    (OUTPUT_DIR / "Substrate_and_Bead_Images" / f"Pass_{pass_number}").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "Bead_Images" / f"Pass_{pass_number}").mkdir(parents=True, exist_ok=True)


# IMAGE EXTENSIONS
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# LOAD CALIBRATION
calibration = SpatialCalibration(show_debug=True)

if CALIBRATION_FILE.exists():

    print("\nLoading existing spatial calibration:")
    print(CALIBRATION_FILE)
    calibration.load_calibration( CALIBRATION_FILE)

else:

    raise FileNotFoundError("Spatial calibration does not exist. "
        "\n Run the calibration step first.")



# PROCESS PASSES
for pass_number, input_dir in PASS_DIRS.items():

    print(f"\n========== PASS {pass_number} ==========")

    output_dir = (OUTPUT_DIR / "Substrate_and_Bead_Images" / f"Pass_{pass_number}")
    image_files = sorted(
        [path
        for path in input_dir.iterdir()
        if path.suffix.lower()
        in IMAGE_EXTENSIONS]
    )

    print(f"Images found: {len(image_files)}")

    for image_path in image_files:

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:

            print(f"Skipping unreadable image: {image_path.name} ")

            continue

    
        ''' Perspective correction
        This output size should eventually be calculated from your 
        physical substrate dimensions and desired m/pixel resolution. '''

        output_width = image.shape[1]
        output_height = image.shape[0]

        transformed = cv2.warpPerspective(image, calibration.H_uv_to_xy, (output_width, output_height))


        # Rotate into simulation orientation
        transformed = cv2.rotate(transformed, cv2.ROTATE_180)

        # ----------------------------------------------------
        # Save without changing original filename
   
        output_path = (output_dir / image_path.name)

        cv2.imwrite(str(output_path), transformed)

    print(f"Pass {pass_number} complete.")