from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ThermalValidationGraphs:

    def __init__(
        self,
        validation,
        substrate_length,
        substrate_width,
        output_dir):

        self.validation = validation
        self.substrate_length = substrate_length
        self.substrate_width = substrate_width
        self.output_dir = Path(output_dir)
        self.create_output_directories()


    # OUTPUT DIRECTORIES
    def create_output_directories( self):

        for pass_number in [1, 2]:
            for folder in [
                "Annotated_Frames",
                "X_Profiles",
                "Y_Profiles",
                "Cooling_Histories"]:

                (self.output_dir
                 / f"Pass_{pass_number}"
                 / folder).mkdir(parents=True, exist_ok=True)

        (self.output_dir / "Summary_Figures").mkdir(parents=True, exist_ok=True)

 
    # COOLING HISTORY

    def plot_cooling_history(self, table, pass_number, block_number, output_path = None):

        data = table[
            (table["pass"] == pass_number) & (table["block"] == block_number)].copy()

        data = data.sort_values("cooling_time_s")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["cooling_time_s"], data["center_T_exp_C"], marker="o", label="FLIR")
        ax.plot( data["cooling_time_s"], data["center_T_sim_C"], marker="s", label="Graph Model")
        ax.set_xlabel("Cooling Time [s]")
        ax.set_ylabel("Center Temperature [°C]")
        ax.set_title(f"Pass {pass_number} - Block {block_number}")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()


        if output_path is None:

            output_path = (self.output_dir
                / f"Pass_{pass_number}"
                / "Cooling_Histories"
                / f"Block_{block_number}.png")

        fig.savefig(output_path, dpi=300)
        plt.close(fig)



    # X TEMPERATURE PROFILE

    def plot_x_profile(
        self,
        experimental_points,
        simulation_points,
        pass_number,
        block_number,
        cooling_time,
        validation_step,
        output_path=None):

        names = ["x_minus", "center", "x_plus"]
        x_exp = np.array([experimental_points[name]["x"] 
                          for name in names])

        T_exp = np.array([experimental_points[name]["temperature"]
                          for name in names])
        x_sim = np.array([
            simulation_points[name]["x"]
            for name in names])


        T_sim = np.array([
            simulation_points[name]["temperature"]
            for name in names])

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot( x_exp * 1000, T_exp, marker="o", label="FLIR")
        ax.plot(x_sim * 1000, T_sim, marker="s", label="Graph Model")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("Temperature [°C]")

        ax.set_title(
            f"Block {block_number} | "
            f"Cooling Time = "
            f"{cooling_time:.3f} s")

        ax.grid(True)
        ax.legend()
        fig.tight_layout()


        if output_path is None:

            output_path = (
                self.output_dir
                / f"Pass_{pass_number}"
                / "X_Profiles"
                / (f"Step_{validation_step}_"
                   f"Block_{block_number}.png")
            )


        fig.savefig( output_path, dpi=300)
        plt.close(fig)


    # Y TEMPERATURE PROFILE

    def plot_y_profile(
        self,
        experimental_points,
        simulation_points,
        pass_number,
        block_number,
        cooling_time,
        validation_step,
        output_path=None):

        names = ["y_minus", "center", "y_plus"]

        y_exp = np.array([
            experimental_points[name]["y"]
            for name in names])

        T_exp = np.array([
            experimental_points[name]["temperature"]
            for name in names])


        y_sim = np.array([
            simulation_points[name]["y"]
            for name in names])

        T_sim = np.array([
            simulation_points[name]["temperature"]
            for name in names])

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot( y_exp * 1000, T_exp, marker="o", label="FLIR")
        ax.plot( y_sim * 1000, T_sim, marker="s", label="Graph Model")
        ax.set_xlabel("y [mm]")
        ax.set_ylabel("Temperature [°C]" )

        ax.set_title(
            f"Block {block_number} | "
            f"Cooling Time = "
            f"{cooling_time:.3f} s")

        ax.grid(True)
        ax.legend()
        fig.tight_layout()


        if output_path is None:

            output_path = (
                self.output_dir
                / f"Pass_{pass_number}"
                / "Y_Profiles"
                / (f"Step_{validation_step}_"
                   f"Block_{block_number}.png")
            )


        fig.savefig(output_path, dpi=300)
        plt.close(fig)


    # PARITY PLOT
 
    def plot_parity( self, table, output_path = None):

        point_names = ["center", "x_minus", "x_plus", "y_minus", "y_plus"]
        T_exp = []
        T_sim = []

        for point in point_names:

            T_exp.extend(table[f"{point}_T_exp_C"].to_numpy())
            T_sim.extend(table[f"{point}_T_sim_C"].to_numpy())

        T_exp = np.array(T_exp)
        T_sim = np.array(T_sim)
        minimum = min(T_exp.min(), T_sim.min())
        maximum = max(T_exp.max(), T_sim.max())
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(T_exp, T_sim, alpha=0.6)


        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
            label="Perfect Agreement")
        ax.set_xlabel("FLIR Temperature [°C]")
        ax.set_ylabel("Graph Model Temperature [°C]")
        ax.set_title("Experimental vs Simulated Temperature")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()


        if output_path is None:

            output_path = (
                self.output_dir
                / "Summary_Figures"
                / "parity_plot.png")

        fig.savefig(output_path, dpi=300)
        plt.close(fig)


    # ========================================================
    # ANNOTATED THERMAL FRAME

    def plot_validation_frame(
        self,
        temperature_field,
        tracked_blocks,
        pass_number,
        frame_count,
        validation_step,
        actual_time = None,
        output_path=None):

        fig, ax = plt.subplots(figsize=(12, 5))
        image = ax.imshow(temperature_field, 
                          cmap="turbo", 
                          origin="lower",
                          extent=[0, self.substrate_length, 0, self.substrate_width ],
                          aspect="equal"
                          )


        for tracked in tracked_blocks:

            block_number = (tracked["block"])
            roi = (self.validation.block_rois[block_number - 1] )
            rectangle = Rectangle(
                (roi["x_min"], roi["y_min"]),
                self.validation.block_length,
                self.validation.block_width,
                fill=False,
                linewidth=2)

            ax.add_patch(rectangle)

            points = (self.validation.create_tracking_points(roi))

            for point_name, (x, y) in points.items():

                ax.scatter(
                    x,
                    y,
                    s=40,
                    edgecolors="black",
                    linewidths=0.75,
                    zorder=3
                )

            ax.text(roi["x_center"],
                    roi["y_max"],

                (f"B{block_number}\n"
                 f"{tracked['cooling_time']:.2f} s"),

                horizontalalignment="center",
                verticalalignment="bottom", 
                fontsize=9
            )


        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if actual_time is None:

            title = (
                f"Pass {pass_number} | "
                f"Validation Step {validation_step} | "
                f"FLIR Frame {frame_count}"
            )

        else:

            title = (
                f"Pass {pass_number} | "
                f"Validation Step {validation_step} | "
                f"t = {actual_time:.3f} s | "
                f"FLIR Frame {frame_count}"
            )


        ax.set_title(
            title
        )


        # ========================================================
        # COLORBAR
        # ========================================================

        fig.colorbar(
            image,
            ax=ax,
            label="Temperature [°C]"
        )


        fig.tight_layout()


        # ========================================================
        # SAVE
        # ========================================================

        if output_path is None:

            output_path = (

                self.output_dir
                / f"Pass_{pass_number}"
                / "Annotated_Frames"
                / (
                    f"Step_{validation_step}_"
                    f"Frame_{frame_count}.png"
                )
            )

        print(
            f"Saved annotated frame: "
            f"{output_path}"
        )

        fig.savefig(
            output_path,
            dpi=300
        )


        plt.close(
            fig
        )


if __name__ == "__main__":


    import os
    import sys
    from pathlib import Path
    from Python.Experimental_Processing.CleanThermalImages.spatial_calibration import SpatialCalibration
    # Get the parent "Python" folder
    python_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # Add it to Python's import search path
    if python_folder not in sys.path:
        sys.path.insert(0, python_folder)

    from GraphCode.config import BUILD, WAAM, BLOCK
    from Python.Experimental_Processing.CleanThermalImages.spatial_calibration import SpatialCalibration
    from Python.Experimental_Processing.CleanThermalImages.thermal_validation import ThermalValidation
    from GraphCode.geometry import Geometry
    
    #Import Geometry
    stl_file = BUILD["stl_file"]
    geom = Geometry(stl_file)
    geom.build_blocks()


    # PATHS
    BASE_DIR = Path(__file__).resolve().parent
    THERMAL_DATA_DIR = BASE_DIR.parent / "ThermalData"
    CALIBRATION_FILE = THERMAL_DATA_DIR / "spatial_calibration.npz"
    DOWNLOADS_DIR = Path(r"C:\Users\Kayleigh\Downloads")
    VALIDATION_DIR = DOWNLOADS_DIR / "Thermal_Validation"

    VALIDATION_TABLE_FILE = (
        VALIDATION_DIR
        / "Tables"
        / "validation_results.csv"
    )

    PROCESSED_DIR = (
        DOWNLOADS_DIR
        / "Thermal_Processed"
        / "Temperature_Data"
    )


    # LOAD SPATIAL CALIBRATION
    calibration =  (SpatialCalibration())
    calibration.load_calibration(CALIBRATION_FILE)


    
    # CREATE VALIDATION OBJECT

    validation = ThermalValidation(
        calibration=calibration,
        blocks_per_layer= BUILD["blocks_per_layer"],
        bead_length= geom.length,
        block_width=BLOCK["width"],
        scan_speed=WAAM["scan_speed"]
    )

   
    # CREATE GRAPHING OBJECT
    graphs = ThermalValidationGraphs(
        validation=validation,
        substrate_length=BUILD["substrate_length"],
        substrate_width=BUILD["substrate_width"],
        output_dir= VALIDATION_DIR
    )

    #---------- Bead Location ------------
    bead_x_start = 12.7/1000

    bead_y_center = (BUILD["substrate_width"] / 2) + 0.001/2

    # Create the five physical block ROIs.
    validation.create_block_rois(bead_x_start=bead_x_start, bead_y_center=bead_y_center)

        # ========================================================
    # PROCESS EACH EXPERIMENTAL PASS
    # ========================================================

    for pass_number in [1, 2]:

        print(
            f"\n======================================"
        )

        print(
            f"Processing Validation Pass "
            f"{pass_number}"
        )

        print(
            f"======================================"
        )


        # ====================================================
        # A700 TIMING CSV
        # ====================================================

        csv_path = (
            THERMAL_DATA_DIR
            / "Processed"
            / f"Pass_{pass_number}"
            / f"A700_Pass{pass_number}.csv"
        )


        print(
            "\nA700 timing file:"
        )

        print(
            csv_path
        )


        timing = (
            validation.load_a700_timing(
                csv_path
            )
        )


        # ====================================================
        # PROCESSED TEMPERATURE DIRECTORY
        # ====================================================

        pass_directory = (
            PROCESSED_DIR
            / f"Pass_{pass_number}"
        )


        # ====================================================
        # VALIDATION TIMESTEPS
        # ====================================================

        for validation_step in range(
            1,
            validation.blocks_per_layer + 1
        ):


            # ------------------------------------------------
            # TARGET MODEL TIME
            # ------------------------------------------------

            target_time = (
                validation_step
                * validation.time_per_block
            )


            print(
                f"\nValidation Step "
                f"{validation_step}"
            )

            print(
                f"Target time = "
                f"{target_time:.3f} s"
            )


            # =================================================
            # FIND NEAREST FLIR FRAME
            # =================================================

            nearest_frame = (
                validation.find_nearest_frame(
                    timing,
                    target_time
                )
            )


            frame_count = int(
                nearest_frame["Count"]
            )


            actual_time = float(
                nearest_frame[
                    "elapsed_time"
                ]
            )


            print(
                f"Selected FLIR frame = "
                f"{frame_count}"
            )

            print(
                f"Actual FLIR time = "
                f"{actual_time:.3f} s"
            )

            print(
                f"Timing error = "
                f"{actual_time - target_time:.4f} s"
            )


            # =================================================
            # FIND PROCESSED TEMPERATURE IMAGE
            # =================================================

            pattern = (
                f"{frame_count:08d}_*.npy"
            )


            matching_files = list(
                pass_directory.glob(
                    pattern
                )
            )


            if len(matching_files) == 0:

                print(
                    f"No processed temperature "
                    f"image found for frame "
                    f"{frame_count}"
                )

                continue


            if len(matching_files) > 1:

                print(
                    f"Warning: multiple files found "
                    f"for frame {frame_count}."
                )

                print(
                    "Using:"
                )

                print(
                    matching_files[0]
                )


            temperature_file = (
                matching_files[0]
            )


            print(
                "Temperature field:"
            )

            print(
                temperature_file
            )


            # =================================================
            # LOAD TEMPERATURE FIELD
            # =================================================

            temperature_field = np.load(
                temperature_file
            )


            print(
                f"Temperature field shape = "
                f"{temperature_field.shape}"
            )


            print(
                f"Temperature range = "
                f"{temperature_field.min():.2f} "
                f"to "
                f"{temperature_field.max():.2f} °C"
            )


            # =================================================
            # DETERMINE PROGRESSIVE BLOCKS
            # =================================================

            tracked_blocks = (
                validation.get_tracked_blocks(
                    validation_step
                )
            )


            print(
                "Tracked blocks:"
            )


            for tracked in tracked_blocks:

                print(
                    f"  Block "
                    f"{tracked['block']}: "
                    f"{tracked['cooling_time']:.3f} s "
                    f"cooling"
                )


            # =================================================
            # PLOT THERMAL IMAGE + BLOCKS + POINTS
            # =================================================

            graphs.plot_validation_frame(

                temperature_field=(
                    temperature_field
                ),

                tracked_blocks=(
                    tracked_blocks
                ),

                pass_number=(
                    pass_number
                ),

                frame_count=(
                    frame_count
                ),

                validation_step=(
                    validation_step
                ),

                actual_time=(
                    actual_time
                )
            )