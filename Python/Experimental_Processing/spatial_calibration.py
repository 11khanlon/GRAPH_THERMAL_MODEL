from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys 

from image_extractor import ImagePointExtractor

sys.path.append(r"C:\\Users\\Kayleigh\\GraphThermalModel\\Python\\GraphCode")

'''
Homography code determines (x, y) <--> (u, v)
Helps map off axis camera orientation to top view 

'''

# ----------------------------------------
class SpatialCalibration():

    def __init__(self, show_debug = False):
        
        self.show_debug = show_debug
        self.point_extractor = ImagePointExtractor()

        #Physical coordinates [x, y]
        self.plate_points_xy = None 

        #camera coordinates 
        self.plate_points_uv = None 

        # Transformation matrices
        self.H_xy_to_uv = None
        self.H_uv_to_xy = None

        #Torch calibration 
        # Polygon vertices relative to torch center [du, dv]
        self.torch_mask_relative_uv = None

        # Center used during torch calibration [u, v]
        self.torch_center_uv = None

        # Physical size represented by one rectified image pixel
        self.pixel_size_x_m = None
        self.pixel_size_y_m = None

        # Inverse spatial resolution
        self.pixels_per_meter_x = None
        self.pixels_per_meter_y = None

        # Rectified image dimensions
        self.rectified_width_px = None
        self.rectified_height_px = None

        # Original camera -> rectified image
        self.H_uv_to_rectified = None


    #-------------------------------------
    # Image loading 

    def load_image(self, image):

        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)

            if image is None:
                raise FileNotFoundError(f"Unable to load image:\n"f"{image}")

        return image

    #----------------------------------
    #Plate calibration 
    def calibrate_plate(self, plate_image, geometry):

        image = self.load_image(plate_image)

        # Physical substrate corners
        self.plate_points_xy = np.array(
            [[geometry.substrate_xmin, geometry.substrate_ymin],
            [geometry.substrate_xmax, geometry.substrate_ymin],
            [geometry.substrate_xmax, geometry.substrate_ymax],
            [geometry.substrate_xmin, geometry.substrate_ymax]], dtype=np.float64)

        print("\nStart bottom left corner and go counter-clockwise")
        print("Select the four physical substrate corners in this order: ")
        print("1 = (xmin, ymin) = (-x, -y)")
        print("2 = (xmax, ymin) = (+x, -y)")
        print("3 = (xmax, ymax) = (+x, +y)")
        print("4 = (xmin, ymax) = (-x, +y)")


        self.plate_points_uv = (
            self.point_extractor.select_points(image,
                window_name = "Select Substrate Corners",
                max_points = 4)
                ).astype(np.float64)

        if len(self.plate_points_uv) != 4:
            raise ValueError("Exactly four substrate corners must be selected")

        # Calculate homography
        self.H_xy_to_uv = (
            cv2.getPerspectiveTransform(
                self.plate_points_xy.astype(np.float32),
                self.plate_points_uv.astype(np.float32))
                )

        self.H_uv_to_xy = np.linalg.inv(self.H_xy_to_uv)

        print("\nH_xy_to_uv:")
        print(self.H_xy_to_uv)

        print("\nH_uv_to_xy:")
        print(self.H_uv_to_xy)

        if self.show_debug:

            self.show_plate_calibration(image)

        return self.H_xy_to_uv
    

    #------------------------------------
    #  Physical [x,y] -> camera [u,v]

    def create_rectified_grid(self):

        if self.plate_points_xy is None:
            raise RuntimeError("Plate calibration must be performed first.")

        #Physical substrate dimension 
        xmin = np.min(self.plate_points_xy[:, 0])
        xmax = np.max(self.plate_points_xy[:, 0])

        ymin = np.min(self.plate_points_xy[:, 1])
        ymax = np.max(self.plate_points_xy[:, 1])

        substrate_length = xmax - xmin
        substrate_width = ymax - ymin

        #Camera Pixel dimensions of substrate 
        p1 = self.plate_points_uv[0]
        p2 = self.plate_points_uv[1]
        p3 = self.plate_points_uv[2]
        p4 = self.plate_points_uv[3]

        # Long substrate edges
        bottom_width_px = np.linalg.norm(p2 - p1)
        top_width_px = np.linalg.norm(p3 - p4)

        # Short substrate edges
        right_height_px = np.linalg.norm(p3 - p2)
        left_height_px = np.linalg.norm(p4 - p1)

        # Average opposite edges
        self.rectified_width_px = int(
            round((bottom_width_px + top_width_px) / 2)
        )

        self.rectified_height_px = int(
            round((right_height_px + left_height_px) / 2)
        )


        # PHYSICAL PIXEL SIZE
        self.pixel_size_x_m = (substrate_length / self.rectified_width_px)
        self.pixel_size_y_m = (substrate_width / self.rectified_height_px)
        self.pixels_per_meter_x = (1.0 / self.pixel_size_x_m)
        self.pixels_per_meter_y = (1.0 / self.pixel_size_y_m)


        # RECTIFIED IMAGE CORNERS
        rectified_points_uv = np.array(
            [[0, 0], 
             [self.rectified_width_px - 1, 0],
             [self.rectified_width_px - 1, self.rectified_height_px - 1],
             [0, self.rectified_height_px - 1]
            ], dtype=np.float32)


        # CAMERA → RECTIFIED HOMOGRAPHY
        self.H_uv_to_rectified = (
            cv2.getPerspectiveTransform(
                self.plate_points_uv.astype(np.float32),
                rectified_points_uv)
                )


        # PRINT RESULTS
        print("\n----------------------------------")
        print("Rectified substrate grid")
        print("----------------------------------")

        print(f"Width: {self.rectified_width_px} pixels ")
        print(f"Height: {self.rectified_height_px} pixels ")
        print(f"x pixel size: {self.pixel_size_x_m * 1000:.4f} mm/pixel" )
        print(f"y pixel size: {self.pixel_size_y_m * 1000:.4f} mm/pixel " )
        print(f"x resolution: {self.pixels_per_meter_x:.2f} pixels/m")
        print(f"y resolution: {self.pixels_per_meter_y:.2f} pixels/m")

        print("\nH_uv_to_rectified:")
        print(self.H_uv_to_rectified)

        return self.H_uv_to_rectified
    

    #------------------------------------
    def rectified_pixel_to_physical(self, u, v, rotated=False):

        if (self.pixel_size_x_m is None
            or self.pixel_size_y_m is None ):

            raise RuntimeError("Rectified grid has not been created." )

        if rotated:
            u = self.rectified_width_px - 1 - u
            v = self.rectified_height_px - 1 - v
        
        xmin = np.min(self.plate_points_xy[:, 0])
        ymin = np.min(self.plate_points_xy[:, 1])

        # Physical coordinate of pixel center
        x = xmin + (u + 0.5)* self.pixel_size_x_m
        y = ymin+ (v + 0.5)* self.pixel_size_y_m
        
        return np.array([x, y], dtype=float)

  

    #--------------------------------
    def save_calibration(self, filepath):

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            filepath,
            plate_points_xy = self.plate_points_xy,  # Selected calibration points
            plate_points_uv = self.plate_points_uv,
            H_xy_to_uv= self.H_xy_to_uv,   # Physical <-> original camera
            H_uv_to_xy= self.H_uv_to_xy,
            H_uv_to_rectified= self.H_uv_to_rectified,    # Original camera -> rectified
            pixel_size_x_m= self.pixel_size_x_m,    # Rectified grid information
            pixel_size_y_m=self.pixel_size_y_m,
            pixels_per_meter_x=self.pixels_per_meter_x,
            pixels_per_meter_y= self.pixels_per_meter_y,
            rectified_width_px= self.rectified_width_px,
            rectified_height_px= self.rectified_height_px
            )

        print(f"\nCalibration saved to: {filepath}" )


    #--------------------------------------
    def load_calibration(self, filepath):

        data = np.load(filepath)
        self.plate_points_xy = data["plate_points_xy"]
        self.plate_points_uv = data["plate_points_uv"]
        self.H_xy_to_uv = data["H_xy_to_uv"]
        self.H_uv_to_xy = data["H_uv_to_xy"]
        self.H_uv_to_rectified = data["H_uv_to_rectified"]

        self.pixel_size_y_m = float(data["pixel_size_y_m"])
        self.pixels_per_meter_x = float(data["pixels_per_meter_x"])
        self.pixels_per_meter_y = float(data["pixels_per_meter_y"] )

        # Rectified image dimensions
        self.rectified_width_px = int( data["rectified_width_px"])
        self.rectified_height_px = int(data["rectified_height_px"])
        print(f"\nCalibration loaded from: {filepath}")
         
        '''self.torch_center_uv = data["torch_center_uv"]
        self.torch_mask_relative_uv = data["torch_mask_relative_uv"]'''

    # --------------------------------------------------------

    def show_plate_calibration(self, image):

        plt.figure(figsize=(10, 7))

        if image.ndim == 2:
            plt.imshow(image, cmap="gray")

        else:
            plt.imshow( cv2.cvtColor(image, cv2.COLOR_BGR2RGB) )

        points = self.plate_points_uv

        # Plot selected corners
        for i, (u, v) in enumerate(points, start=1):

            plt.scatter(u, v)
            plt.annotate( str(i), (u, v), fontsize=12)


        # Close polygon
        closed_points = np.vstack([points, points[0]] )
        plt.plot(closed_points[:, 0], closed_points[:, 1])
        plt.xlabel( "u [pixels]")
        plt.ylabel( "v [pixels]")
        plt.title( "Selected Substrate Calibration")
        plt.show()



if __name__ == "__main__":

    #Files
    calibration_image = Path(
        r"C:\\Users\\Kayleigh\\GraphThermalModel\\Python\\ThermalData\\Substrate_Calibration.tiff"
    )

    calibration_file = Path(
        r"C:\\Users\\Kayleigh\\GraphThermalModel\\Python\\ThermalData\\spatial_calibration.npz"
    )

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


    # KNOWN SUBSTRATE GEOMETRY
    class CalibrationGeometry:

        substrate_xmin = 0.0
        substrate_xmax = BUILD["substrate_length"]

        substrate_ymin = 0.0
        substrate_ymax = BUILD["substrate_width"]

    geometry = CalibrationGeometry()

    # INITIALIZE
    calibration = SpatialCalibration(show_debug=True)

    # LOAD IMAGE
    image = calibration.load_image(calibration_image)
    print("\nCalibration image:")
    print( f"Shape = {image.shape}" )
    print(f"dtype = {image.dtype}" )
    

    # STEP 1: Select four substrate corners
    calibration.calibrate_plate(
        plate_image=image,
        geometry=geometry)

    # STEP 2: Build rectified substrate grid
    calibration.create_rectified_grid()

    # STEP 3: Save everything
    calibration.save_calibration( calibration_file)
    print( "\nSpatial calibration complete.")