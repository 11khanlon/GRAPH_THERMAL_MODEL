import numpy as np
import pandas as pd
from pathlib import Path
import sys

class ThermalValidation:

    def __init__(
        self,
        calibration,
        blocks_per_layer,
        bead_length,
        block_width,
        scan_speed):

        self.calibration = calibration
        self.blocks_per_layer = blocks_per_layer
        self.bead_length = bead_length
        self.block_width = block_width
        self.scan_speed = scan_speed

        self.block_length = bead_length / blocks_per_layer
        self.time_per_block = self.block_length / scan_speed
        self.block_rois = []


    def load_a700_timing(self, csv_path):

        timing = pd.read_csv(csv_path)
        print("\nA700 CSV columns:")
        print( timing.columns)

        # CONVERT TIMESTAMP
        timing["Timestamp"] = pd.to_datetime(timing["Timestamp"])
      
        # FIND FIRST DEPOSITION FRAME
        # IsDepositing may be read as bool, integer, or string. Convert it robustly to a boolean mask.

        deposition_mask = (
            timing["IsDepositing"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        deposition_rows = timing[deposition_mask]

        if deposition_rows.empty:
            raise ValueError("No deposition frames were found in the A700 timing CSV. ")


        # First FLIR frame where deposition begins
        deposition_start_index = (deposition_rows.index[0])
        deposition_start_time = (timing.loc[deposition_start_index, "Timestamp"])
        deposition_start_count = int(timing.loc[deposition_start_index, "Count"])


        # DEFINE t = 0 AT DEPOSITION START
        timing["elapsed_time"] = (timing["Timestamp"] - deposition_start_time).dt.total_seconds()


        # PRINT SYNCHRONIZATION INFORMATION
        print("\nDeposition start:")
        print(f"Count = {deposition_start_count} ")
        print(f"Timestamp = {deposition_start_time}")
        print(f"Stage = {timing.loc[deposition_start_index, 'Stage']} ")
        print(f"Position = {timing.loc[deposition_start_index, 'Position_m']:.6f} m")

      
        return timing


    def create_validation_times(self):

        validation_times = np.arange(self.blocks_per_layer + 1) * self.time_per_block

        return validation_times


    def find_nearest_frame(self, timing, target_time):

        index = (timing["elapsed_time"] - target_time).abs().idxmin()

        return timing.loc[index]


    def create_block_rois(self, bead_x_start, bead_y_center):

        self.block_rois = []

        for block_index in range(self.blocks_per_layer):

            x_min = bead_x_start + block_index * self.block_length
            x_max = x_min + self.block_length

            y_min = bead_y_center - self.block_width / 2
            y_max =  bead_y_center + self.block_width / 2
            roi = {
                "block": block_index + 1,

                "x_min": x_min,
                "x_max": x_max,

                "y_min": y_min,
                "y_max": y_max,

                "x_center": (x_min + x_max) / 2,
                "y_center": bead_y_center
                }

            self.block_rois.append(roi)

        return self.block_rois
    

    def get_tracked_blocks(self, validation_step):

        tracked = []

        for block_number in range(1, validation_step + 1):

            cooling_steps = validation_step - block_number + 1
            cooling_time = cooling_steps * self.time_per_block
            tracked.append(
                {"block": block_number,
                 "cooling_steps": cooling_steps,
                 "cooling_time": cooling_time}
            )

        return tracked
    

    def create_tracking_points(self, roi):

        x_center = roi["x_center"]
        y_center = roi["y_center"]

        dx = self.block_length / 4
        dy = self.block_width / 4

        points = {
            "center": (x_center, y_center),
            "x_minus": (x_center - dx, y_center),
            "x_plus": (x_center + dx, y_center),
            "y_minus": (x_center, y_center - dy),
            "y_plus": (x_center, y_center + dy)
            }

        return points


    def physical_to_rectified_pixel(self, x, y, rotated=True):

        xmin = np.min(self.plate_points_xy[:, 0])
        ymin = np.min(self.plate_points_xy[:, 1])
        u =  (x - xmin) / self.pixel_size_x_m
        v = (y - ymin) / self.pixel_size_y_m
        
        if rotated:

            u = self.rectified_width_px - 1 - u
            v = self.rectified_height_px - 1 - v

        return (int(round(u)), int(round(v)))


    def extract_point_temperatures(self, temperature_field, roi):

        points = self.create_tracking_points(roi)
        temperatures = {}

        for name, (x, y) in points.items():

            u, v = self.calibration.physical_to_rectified_pixel(x, y, rotated=True)
            temperature = temperature_field[v, u]
            temperatures[name] = {
                "x": x,
                "y": y,
                "u": u,
                "v": v,
                "temperature": float(temperature)
            }

        return temperatures

      
    # TEMPERATURE GRADIENTS
    def calculate_gradients(self, points):

        T_x_minus = points["x_minus"]["temperature"]
        T_x_plus = points["x_plus"]["temperature"]
        T_y_minus = points["y_minus"]["temperature"]
        T_y_plus = points["y_plus"]["temperature"]
        
        x_minus = points["x_minus"]["x"]
        x_plus =  points["x_plus"]["x"]
        y_minus = points["y_minus"]["y"]
        y_plus = points["y_plus"]["y"]
        
        dTdx = (T_x_plus - T_x_minus) / (x_plus - x_minus)   #Gradient x
        dTdy = (T_y_plus - T_y_minus) / (y_plus - y_minus)   #Gradient y
        
        return { "dTdx_C_per_m": dTdx, "dTdy_C_per_m": dTdy}

    
    def extract_cooling_history(self, timing,pass_directory, pass_number):
        records = []

        # Pre-index temperature files by FLIR frame number
        frame_files = {}

        for file in Path(pass_directory).glob("*.npy"):
            try:
                frame_number = int(file.stem.split("_")[0])
                frame_files[frame_number] = file
            except ValueError:
                continue

        # Process frames in chronological order
        timing_sorted = timing.sort_values("elapsed_time")

        for _, frame in timing_sorted.iterrows():

            frame_count = int(frame["Count"])
            actual_time = float(frame["elapsed_time"])

            if frame_count not in frame_files:
                continue

            # Number of blocks that have completed deposition
            completed_blocks = min(
                self.blocks_per_layer,
                int(np.floor(actual_time / self.time_per_block))
            )

            if completed_blocks < 1:
                continue

            temperature_field = np.load(frame_files[frame_count])

            for block_number in range(1, completed_blocks + 1):
                roi = self.block_rois[block_number - 1]
                points = self.extract_point_temperatures(temperature_field, roi)
                block_end_time = block_number * self.time_per_block

                record = {
                    "pass": pass_number,
                    "frame_count": frame_count,
                    "block": block_number,
                    "time_s": actual_time,
                    "actual_cooling_time_s": (actual_time - block_end_time
                    )
                }

                for name, point in points.items():
                    record[f"{name}_T_exp_C"] = point["temperature"]

                gradients = self.calculate_gradients(points)
                record["dTdx_exp_C_per_m"] = gradients["dTdx_C_per_m"]
                record["dTdy_exp_C_per_m"] =  gradients["dTdy_C_per_m"]
                records.append(record)

        history = pd.DataFrame(records)

        if history.empty:
            return history

        history = history.sort_values(["pass", "block", "actual_cooling_time_s"]).reset_index(drop=True)

        # Cooling rate at each tracking point
        for point_name in ["center", "x_minus", "x_plus","y_minus", "y_plus"]:
            temperature_column = f"{point_name}_T_exp_C"
            history[f"{point_name}_dTdt_C_per_s"] = (history.groupby(["pass", "block"])[temperature_column].diff()/history.groupby(["pass", "block"])["actual_cooling_time_s"].diff())

        return history



    # MODEL COMPARISON

    def compare_block(
        self,
        pass_number,
        frame_count,
        validation_step,
        block_number,
        cooling_time,
        experimental_points,
        simulation_points):

        row = {
            "pass": pass_number,
            "frame_count": frame_count,
            "validation_step": validation_step,
            "block": block_number,
            "cooling_time_s": cooling_time
        }

        point_names = ["center", "x_minus", "x_plus", "y_minus", "y_plus"]
        errors = []

        # Compare temperatures at all five points
        for point_name in point_names:

            exp = experimental_points[point_name]
            sim = simulation_points[point_name]
            T_exp = exp["temperature"]
            T_sim = sim["temperature"]
            error = T_sim - T_exp
            abs_error = abs(error)
            row[f"{point_name}_x_m"] = exp["x"]
            row[f"{point_name}_y_m"] = exp["y"]

            row[f"{point_name}_T_exp_C"] = T_exp
            row[f"{point_name}_T_sim_C"] = T_sim
            row[f"{point_name}_error_C"] = error
            row[f"{point_name}_abs_error_C"] = abs_error

            errors.append(error)


        # Overall temperature errors for this block/time

        errors = np.array(errors)
        row["MAE_C"] = np.mean(np.abs(errors))
        row["RMSE_C"] = np.sqrt(np.mean(errors**2))
        row["max_abs_error_C"] = np.max(np.abs(errors))


        # Experimental temperature gradients
        exp_gradients = self.calculate_gradients(experimental_points)

        # Simulated temperature gradients
        sim_gradients = self.calculate_gradients(simulation_points)

        # X gradient comparison
        row["dTdx_exp_C_per_m"] = (exp_gradients["dTdx_C_per_m"])
        row["dTdx_sim_C_per_m"] = (sim_gradients["dTdx_C_per_m"])
        row["dTdx_error_C_per_m"] = (row["dTdx_sim_C_per_m"] - row["dTdx_exp_C_per_m"])

    
        # Y gradient comparison
        row["dTdy_exp_C_per_m"] = (exp_gradients["dTdy_C_per_m"])
        row["dTdy_sim_C_per_m"] = (sim_gradients["dTdy_C_per_m"])
        row["dTdy_error_C_per_m"] = (row["dTdy_sim_C_per_m"] - row["dTdy_exp_C_per_m"])

        return row


    # VALIDATION TABLE
    def create_experimental_table(self, experimental_records):

        table = pd.DataFrame(experimental_records)

        if table.empty:
            print("No experimental records were provided.")
            return table

        table = table.sort_values(by=["pass", "validation_step", "block"])
        table = table.reset_index(drop=True)

        return table


    def save_experimental_table(self, table, output_path):

        output_path = Path(output_path)
        output_path.parent.mkdir( parents=True, exist_ok=True)
        table.to_csv( output_path, index=False)

        print(f"Experimental table saved to: {output_path}")


    
    def create_experimental_record(
        self,
        pass_number,
        frame_count,
        validation_step,
        block_number,
        cooling_time,
        experimental_points,
        actual_time=None,
        temperature_field=None
    ):
        """
        Create one experimental observation for one block
        at one FLIR frame.

        Includes:
        - Frame and timing information
        - Five tracking-point temperatures
        - Physical and pixel coordinates
        - X/Y temperature gradients
        - ROI temperature statistics
        - ROI maximum temperature and its location
        """

        row = {
            "pass": int(pass_number),
            "frame_count": int(frame_count),
            "validation_step": int(validation_step),
            "block": int(block_number),
            "time_s": float(actual_time) if actual_time is not None else np.nan,
            "cooling_time_s": float(cooling_time),
            "nominal_time_s": (
                float(validation_step * self.time_per_block)
            ),
            "frame_time_error_s": (
                float(actual_time - validation_step * self.time_per_block)
                if actual_time is not None else np.nan
            )
        }

        roi = self.block_rois[block_number - 1]


        block_end_time = block_number * self.time_per_block

        row["block_end_time_s"] = float(block_end_time)

        row["actual_cooling_time_s"] = (
            float(actual_time - block_end_time)
            if actual_time is not None else np.nan
        )

        row["nominal_cooling_since_completion_s"] = (
            float(validation_step * self.time_per_block - block_end_time)
        )

        # --------------------------------------------------
        # ROI physical boundaries
        # --------------------------------------------------

        for key in ("x_min", "x_max", "y_min", "y_max", "x_center", "y_center"):
            row[f"roi_{key}_m"] = float(roi[key])

        # --------------------------------------------------
        # Five tracking points
        # --------------------------------------------------

        point_names = [
            "center", "x_minus", "x_plus",
            "y_minus", "y_plus"
        ]

        for name in point_names:
            point = experimental_points[name]

            row[f"{name}_x_m"] = float(point["x"])
            row[f"{name}_y_m"] = float(point["y"])
            row[f"{name}_u_px"] = int(point["u"])
            row[f"{name}_v_px"] = int(point["v"])
            row[f"{name}_T_exp_C"] = float(point["temperature"])

        # --------------------------------------------------
        # Spatial temperature gradients
        # --------------------------------------------------

        gradients = self.calculate_gradients(experimental_points)

        row["dTdx_exp_C_per_m"] = float(
            gradients["dTdx_C_per_m"]
        )
        row["dTdy_exp_C_per_m"] = float(
            gradients["dTdy_C_per_m"]
        )

        # --------------------------------------------------
        # FLIR temperature statistics inside the block ROI
        # --------------------------------------------------

        if temperature_field is not None:

            corners = [
                (roi["x_min"], roi["y_min"]),
                (roi["x_min"], roi["y_max"]),
                (roi["x_max"], roi["y_min"]),
                (roi["x_max"], roi["y_max"])
            ]

            pixels = [
                self.calibration.physical_to_rectified_pixel(
                    x, y, rotated=True
                )
                for x, y in corners
            ]

            u_values = [p[0] for p in pixels]
            v_values = [p[1] for p in pixels]

            height, width = temperature_field.shape

            u0 = max(0, min(u_values))
            u1 = min(width, max(u_values) + 1)
            v0 = max(0, min(v_values))
            v1 = min(height, max(v_values) + 1)

            roi_field = temperature_field[v0:v1, u0:u1]

            if roi_field.size == 0 or not np.any(np.isfinite(roi_field)):
                raise ValueError(
                    f"No valid FLIR pixels for block {block_number}"
                )

            row["roi_pixel_count"] = int(
                np.count_nonzero(np.isfinite(roi_field))
            )
            row["roi_T_min_C"] = float(np.nanmin(roi_field))
            row["roi_T_mean_C"] = float(np.nanmean(roi_field))
            row["roi_T_max_C"] = float(np.nanmax(roi_field))
            row["roi_T_std_C"] = float(np.nanstd(roi_field))

            peak_index = np.nanargmax(roi_field)
            peak_v, peak_u = np.unravel_index(
                peak_index, roi_field.shape
            )

            row["roi_T_max_u_px"] = int(u0 + peak_u)
            row["roi_T_max_v_px"] = int(v0 + peak_v)

        return row


    

if __name__ == "__main__":

    import os
    from spatial_calibration import SpatialCalibration
    import sys
    from pathlib import Path


    from pathlib import Path
    import sys

    # ----- Project directories -----


    BASE_DIR = Path(__file__).resolve().parent

    PROJECT_DIR = BASE_DIR.parent.parent.parent
    PYTHON_DIR = BASE_DIR.parent.parent
    GRAPH_CODE_DIR = PYTHON_DIR / "GraphCode"

    sys.path.insert(0, str(PYTHON_DIR))
    sys.path.insert(0, str(GRAPH_CODE_DIR))

    from GraphCode.config import BUILD, BLOCK, MATERIAL, WAAM, IMAGE_PROCESSING
    from GraphCode.geometry import Geometry

    #----- Paths -----
    THERMAL_DATA_DIR = PROJECT_DIR / "A36Steel_Info" / "ThermalData"
    CALIBRATION_FILE = THERMAL_DATA_DIR / "spatial_calibration.npz"

    DOWNLOADS_DIR = Path(r"C:\Users\Kayleigh\Downloads")
    VALIDATION_DIR = DOWNLOADS_DIR / "Thermal_Validation"
    VALIDATION_TABLE = VALIDATION_DIR / "Tables" / "validation_results.csv"
    

    #----- Import File Path ------
    calibration_file = Path( r"C:\\Users\\Kayleigh\\GraphThermalModel\\A36Steel_Info\\ThermalData\\spatial_calibration.npz")



    # ----- Import paths -----
    sys.path.insert(0, str(PYTHON_DIR))

    from GraphCode.config import BUILD, MATERIAL, WAAM, IMAGE_PROCESSING,BLOCK
    from GraphCode.geometry import Geometry
    from GraphCode.material import WireMaterial, SubstrateMaterial

    substrate_material, wire_material = SubstrateMaterial, WireMaterial
    stl_file = BUILD["stl_file"]
    geom = Geometry(stl_file, wire_material, substrate_material)
    geom.build_blocks()

    calibration = SpatialCalibration()
    calibration.load_calibration(calibration_file)


    # CREATE THERMAL VALIDATION OBJECT
    validation = ThermalValidation(
        calibration = calibration,
        blocks_per_layer = BUILD["blocks_per_layer"],
        bead_length = geom.length,
        block_width  = BLOCK["width"],
        scan_speed = WAAM["scan_speed"]
    )


    # PRINT VALIDATION PARAMETERS
    print("\nThermal Validation Parameters")
    print(f"Blocks per layer: {validation.blocks_per_layer}")
    print(f"Bead length: {validation.bead_length:.6f} m")
    print(f"Block length: {validation.block_length:.6f} m")
    print(f"Block width: {validation.block_width:.6f} m ")
    print(f"Scan speed: {validation.scan_speed:.6f} m/s ")
    print(f"Time per block: {validation.time_per_block:.6f} s ")

  
    # CREATE MODEL VALIDATION TIMES
    validation_times =  validation.create_validation_times()
    print("\n Validation times [s]:")
    print(validation_times)



     
