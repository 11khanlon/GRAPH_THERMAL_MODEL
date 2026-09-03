from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys 

from image_extractor import ImagePointExtractor

sys.path.append(r"C:\Users\Kayleigh\GraphThermalModel\Python\GraphCode")


# ----------------------------------------
class ThermalCameraCalibration():

    def __init__(self, show_debug = False):
        
        self.show_debug = show_debug
        self.point_extractor = ImagePointExtractor()

        #Physical coordinates [x, y]
        self.plate_points_xy = None 

        #camera coordinates 
        self.plates_points_uv = None 

        # Transformation matrices
        self.H_xy_to_uv = None
        self.H_uv_to_xy = None

        #Torch calibration 
        # Polygon vertices relative to torch center [du, dv]
        self.torch_mask_relative_uv = None

        # Center used during torch calibration [u, v]
        self.torch_center_uv = None

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

        print("\nSelect the four physical substrate corners in this order: ")
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

    def physical_to_pixel(self, x, y):

        if self.H_xy_to_uv is None:
            raise RuntimeError("Plate calibration has not been performed.")

        point_xy_h = np.array([x, y, 1.0], dtype=float)
        point_uv_h = self.H_xy_to_uv @ point_xy_h
        point_uv_h /= point_uv_h[2]

        return point_uv_h[:2]

    #--------------------------------
    # Camera [u,v] -> physical [x,y] 
    def pixel_to_physical(self, u, v):

        if self.H_uv_to_xy is None:
            raise RuntimeError("Plate calibration has not been performed.")

        point_uv_h = np.array([u, v, 1.0], dtype = float)
        point_xy_h = self.H_uv_to_xy @ point_uv_h
        point_xy_h /= point_xy_h[2]
        
        return point_xy_h[:2]


    #---------------------------------------
    #Define a polygon around the torch/nozzle

    def calibrate_torch(self, torch_image, center_xy = None):

        image = self.load_image(torch_image)

        #Select torch outline
        print("\nSelect points around the torch/nozzle.")
        print("Use enough points to describe the occluded region, then press Enter.")   
        torch_points_uv = (
            self.point_extractor.select_points(
                image, window_name="Select Torch Boundary")
                ).astype(float)


        if len(torch_points_uv) < 3:
            raise ValueError("At least three points are required to define a torch polygon ")

        # Determine torch center
        if center_xy is None:
            print("\nSelect the approximate torch center.")

            center_points = (
                self.point_extractor.select_points(
                    image, window_name="Select Torch Center", max_points = 1)
                    )

            if len(center_points) != 1:
                raise ValueError("Exactly one torch center must be selected ")
            center_uv = center_points[0].astype(float)

        else:
            if self.H_xy_to_uv is None:
                raise RuntimeError("Plate calibration must be performed before using a physical torch center.")

            x, y = center_xy
            center_uv = self.physical_to_pixel(x, y)

        self.torch_center_uv = center_uv

        
        # Store mask relative to torch center
        self.torch_mask_relative_uv = torch_points_uv - center_uv
    
        if self.show_debug:
            self.show_torch_calibration(image, torch_points_uv, center_uv)

        return self.torch_mask_relative_uv

        
    # ========================================================
    # TORCH MASK OPERATIONS
    # ========================================================

    #------- Translate the torch mask to new camera position -------
    def move_torch_mask(self, center_uv):

        if (self.torch_mask_relative_uv is None):

            raise RuntimeError("Torch has not been calibrated.")

        center_uv = np.asarray(center_uv, dtype = float)
        polygon_uv = (self.torch_mask_relative_uv + center_uv)

        return polygon_uv


    #------- Generate binary torch mask ---------
    def create_torch_mask(self, image_shape,center_uv):

        polygon_uv = (self.move_torch_mask(center_uv))
        mask = np.zeros(image_shape[:2], dtype = np.uint8)

        cv2.fillPoly(mask,
            [np.round(polygon_uv).astype(np.int32)], 1)

        return mask


    # --------- Save and Load Calibration -----------
    def save_calibration(self,filepath):

        filepath = Path(filepath)
        filepath.parent.mkdir(parents = True, exist_ok = True)

        np.savez_compressed(
            filepath,
            plate_points_xy = self.plate_points_xy,
            plate_points_uv = self.plate_points_uv,
            H_xy_to_uv = self.H_xy_to_uv,
            H_uv_to_xy = self.H_uv_to_xy,
            torch_center_uv = self.torch_center_uv,
            torch_mask_relative_uv = self.torch_mask_relative_uv)

        print(f"Calibration saved to:\n" f"{filepath}")


    # --------------------------------------------------------

    def load_calibration(self, filepath):

        data = np.load(filepath, allow_pickle = True)

        self.plate_points_xy = data["plate_points_xy"]
        self.plate_points_uv = data["plate_points_uv"]
        self.H_xy_to_uv = data["H_xy_to_uv"]
        self.H_uv_to_xy = data["H_uv_to_xy"]
        self.torch_center_uv = data["torch_center_uv"]
        self.torch_mask_relative_uv = data["torch_mask_relative_uv"]
        


    # --------------------------------------------------------

    def show_plate_calibration(self, image):

        plt.figure(figsize = (8, 6))

        if image.ndim == 2:
            plt.imshow(image, cmap="gray")

        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


        for i, (u, v) in enumerate(self.plate_points_uv, start = 1):

            plt.scatter(u, v)
            plt.annotate(str(i), (u, v))

        plt.xlabel("u [pixels]")
        plt.ylabel("v [pixels]")
        plt.title("Plate Spatial Calibration")
        plt.show()


    # --------------------------------------------------------

    def show_torch_calibration(self, image, torch_points_uv, center_uv):

        display_image = image.copy()
        polygon = np.round(torch_points_uv).astype(np.int32)
        cv2.polylines(display_image, [polygon], isClosed = True, color = 255, thickness = 2)

        center_u = int(round(center_uv[0]))
        center_v = int(round(center_uv[1]))

        cv2.circle(display_image,
            (center_u, center_v), radius = 5, color = 255, thickness = -1)

        cv2.namedWindow("Torch Calibration", cv2.WINDOW_NORMAL)
        cv2.imshow("Torch Calibration", display_image)
        cv2.waitKey(0)
        cv2.destroyWindow("Torch Calibration")



if __name__ == "__main__":

    print("Testing thermal camera calibration...")
    calibration = (ThermalCameraCalibration(show_debug=True))

    '''x_test = 0.010
    y_test = 0.002

    u, v = (calibration.physical_to_pixel(x_test, y_test))
    x_back, y_back = (calibration.pixel_to_physical(u, v))

    print("\nCoordinate round-trip test:")
    print(f"Original xy: "
        f"({x_test:.6f}, "
        f"{y_test:.6f})")

    print(f"Camera uv: "
        f"({u:.3f}, "
        f"{v:.3f})")

    print(f"Recovered xy: "
        f"({x_back:.6f}, "
        f"{y_back:.6f})")

    error = np.linalg.norm(np.array([x_test, y_test]) - np.array([x_back, y_back]))
    print(f"Round-trip error: " f"{error:.3e} m")

    calibration.torch_mask_relative_uv = (
    np.array([
        [-10, -20],
        [ 10, -20],
        [ 10,  20],
        [-10,  20]
        ], dtype=float)
    )

    new_center_uv = np.array([300, 200])
    polygon_uv = (calibration.move_torch_mask(new_center_uv))

    print("\nMoved torch polygon:")
    print(polygon_uv)

    mask = (calibration.create_torch_mask(image_shape=(480,640), 
                                          center_uv=new_center_uv))

    print("Mask shape:", mask.shape)
    print("Masked pixels:", np.sum(mask))

    assert ( mask.shape == (480,640))

    test_file = (
    Path("test_camera_calibration.npz")
    )

    calibration.save_calibration(
        test_file
    )

    loaded = (
        ThermalCameraCalibration()
    )

    loaded.load_calibration(
        test_file
    )

    assert np.allclose(
        calibration.torch_mask_relative_uv,
        loaded.torch_mask_relative_uv
    )

    assert np.allclose(
        calibration.H_xy_to_uv,
        loaded.H_xy_to_uv
    )
        

    # Add a temporary image path here
    # when testing interactively.'''
    print("Calibration class initialized successfully.")