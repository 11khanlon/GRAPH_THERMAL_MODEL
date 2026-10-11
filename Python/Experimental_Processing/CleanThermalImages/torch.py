from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys 

from Python.Experimental_Processing.CleanThermalImages.image_extractor import ImagePointExtractor


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
