from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


class TemperatureConverter:

    def __init__(self):

        self.Tmin = None
        self.Tmax = None

    # Extract Tmin and Tmax from old A700 filename
    def get_temperature_range(self, image_path):

        image_path = Path(image_path)
        stem = image_path.stem
        parts = stem.split("_")

        if len(parts) < 3:
            raise ValueError("Expected filename format: COUNT_TMIN_TMAX.tiff")

        self.Tmin = float(parts[-2])
        self.Tmax = float(parts[-1])
        return  self.Tmin, self.Tmax

        

    # Convert saved uint16 intensity back to °C
    
    def intensity_to_temperature(self, image_16bit, Tmin=None, Tmax=None):

        if Tmin is None:
            Tmin = self.Tmin

        if Tmax is None:
            Tmax = self.Tmax

        if Tmin is None or Tmax is None:
            raise RuntimeError("Tmin and Tmax have not been defined.")
              
        U = image_16bit.astype(np.float64)
        temperature = Tmin + (U / 65535.0)* (Tmax - Tmin)
    
        return temperature


if __name__ == "__main__":
   
    # FILES
    from Python.Experimental_Processing.CleanThermalImages.spatial_calibration import SpatialCalibration 

    image_path = Path(r"C:\Users\Kayleigh\Downloads\Thermal_Images_Pass1\00001064_21.46_521.78.tiff")

    calibration_path = Path( r"C:\Users\Kayleigh\GraphThermalModel\Python\ThermalData\spatial_calibration.npz")

    import os
    import sys
    
    # Get the parent "Python" folder
    python_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # Add it to Python's import search path
    if python_folder not in sys.path:
        sys.path.insert(0, python_folder)

    from GraphCode.config import BUILD

    # LOAD TIFF
    image_16bit = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image_16bit is None:
        raise FileNotFoundError(f"Could not load:\n{image_path}")
    
    # TIFF INTENSITY -> TEMPERATURE
    converter = TemperatureConverter()

    Tmin, Tmax = (converter.get_temperature_range(image_path))
    temperature = (converter.intensity_to_temperature(image_16bit))
    print(f"\nTmin from filename = {Tmin:.2f} °C")
    print(f"Tmax from filename = {Tmax:.2f} °C") 
    print(f"Reconstructed min = {temperature.min():.2f} °")
    print(f"Reconstructed max = {temperature.max():.2f} °C") 
    print(f"Original temperature field = {temperature.shape}")
     

    # LOAD SPATIAL CALIBRATION
    calibration = SpatialCalibration()
    calibration.load_calibration(calibration_path)


    # RECTIFY TEMPERATURE FIELD
   
    temperature_rectified = (
        cv2.warpPerspective(temperature.astype(np.float32),
            calibration.H_uv_to_rectified,
            (calibration.rectified_width_px, calibration.rectified_height_px),
            flags=cv2.INTER_LINEAR)
            )


    print(f"Rectified temperature field = {temperature_rectified.shape}") 

    # DISPLAY
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Original camera perspective
    im1 = axes[0].imshow(temperature, cmap="turbo", vmin=Tmin, vmax=Tmax)

    axes[0].set_title("Original A700 Temperature Field")
    axes[0].set_xlabel("u [pixels]")
    axes[0].set_ylabel("v [pixels]")

    fig.colorbar(im1, ax=axes[0], label="Temperature [°C]")

    # Rectified physical substrate
    im2 = axes[1].imshow(
        temperature_rectified,
        cmap="turbo",
        origin="lower",
        extent = [0, BUILD["substrate_length"]*1000, 0, BUILD["substrate_width"]*1000],
        vmin=Tmin,
        vmax=Tmax,
        aspect="equal")

    axes[1].set_title("Rectified Substrate Temperature Field")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")

    fig.colorbar(im2, ax=axes[1], label="Temperature [°C]")


    plt.tight_layout()
    plt.show()