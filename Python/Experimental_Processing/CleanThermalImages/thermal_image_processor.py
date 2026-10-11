from ast import Import
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys 
import os


from Python.Experimental_Processing.CleanThermalImages.spatial_calibration import SpatialCalibration
from Python.Experimental_Processing.CleanThermalImages.temperature_conversion import TemperatureConverter

python_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Add it to Python's import search path
if python_folder not in sys.path:
    sys.path.insert(0, python_folder)

from GraphCode.config import BUILD



# PATHS
BASE_DIR = Path(__file__).resolve().parent
THERMAL_DATA_DIR = (BASE_DIR.parent / "ThermalData")

CALIBRATION_IMAGE = (
    THERMAL_DATA_DIR
    / "Substrate_Calibration.tiff")

CALIBRATION_FILE = (
    THERMAL_DATA_DIR
    / "spatial_calibration.npz")

DOWNLOADS_DIR = Path(r"C:\Users\Kayleigh\Downloads")

PASS_DIRS = {
    1: Path( r"C:\Users\Kayleigh\Downloads\Thermal_Images_Pass1"),
    2: Path(r"C:\Users\Kayleigh\Downloads\Thermal_Images_Pass2")
}

OUTPUT_DIR = (
    DOWNLOADS_DIR
    / "Thermal_Processed")


# ============================================================
# OUTPUT DIRECTORIES -- Need to fix this 

for pass_number in [1, 2]:
    (OUTPUT_DIR / "Temperature_Data" / f"Pass_{pass_number}").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "Temperature_Images" / f"Pass_{pass_number}").mkdir(parents=True, exist_ok=True)


# IMAGE EXTENSIONS
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# LOAD CALIBRATION
calibration = SpatialCalibration()

if CALIBRATION_FILE.exists():

    print("\nLoading spatial calibration:")
    print(CALIBRATION_FILE)

    calibration.load_calibration(CALIBRATION_FILE)

else:

    raise FileNotFoundError(
        "Spatial calibration does not exist.\n"
        "Run spatial_calibration.py first.")

temperature_converter = (TemperatureConverter())


# PROCESS PASSES
for pass_number, input_dir in PASS_DIRS.items():

    print(f"\n========== PASS {pass_number} ==========")

    data_output_dir = (
        OUTPUT_DIR
        / "Temperature_Data"
        / f"Pass_{pass_number}"
    )

    image_output_dir = (
        OUTPUT_DIR
        / "Temperature_Images"
        / f"Pass_{pass_number}"
    )

    image_files = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]
    )

    print(f"Images found: {len(image_files)}")
    print(f"Data output: {data_output_dir}")
    print(f"Image output: {image_output_dir}")


    for image_path in image_files:

        # 1. LOAD ORIGINAL UINT16 TIFF
        image_16bit = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image_16bit is None:

            print(f"Skipping unreadable image: {image_path.name}")

            continue

        # 2. GET Tmin / Tmax FROM FILENAME
        Tmin, Tmax = (temperature_converter.get_temperature_range(image_path))


        # 3. UINT16 -> TEMPERATURE [°C]
        temperature = (temperature_converter.intensity_to_temperature(image_16bit, Tmin, Tmax))


        # 4. CAMERA -> RECTIFIED SUBSTRATE
        temperature_rectified = (cv2.warpPerspective(temperature.astype(np.float32),
                                calibration.H_uv_to_rectified,
                                (calibration.rectified_width_px, calibration.rectified_height_px),
                                flags=cv2.INTER_LINEAR
                                )
                            )

        # 5. ROTATE INTO SIMULATION ORIENTATION
        temperature_rectified = (cv2.rotate(temperature_rectified, cv2.ROTATE_180))


    
        # 6. SAVE NUMERICAL TEMPERATURE FIELD
        output_path = (data_output_dir
                       / f"{image_path.stem}.npy"
                    )

        np.save(output_path, temperature_rectified)

      
        # 7. SAVE RAINBOW TEMPERATURE IMAGE
        figure_path = (image_output_dir
                       / f"{image_path.stem}.png")

        fig, ax = plt.subplots(figsize=(12, 4))
        image_plot = ax.imshow(temperature_rectified, 
                               cmap="turbo", 
                               origin="lower",
                               extent=[0, BUILD["substrate_length"], 0, BUILD["substrate_width"]],
                               vmin=Tmin, vmax=Tmax, aspect="equal")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Pass {pass_number} | {image_path.stem} ")

        fig.colorbar(image_plot, ax=ax, label="Temperature [°C]")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)


    print(f"Pass {pass_number} complete.")