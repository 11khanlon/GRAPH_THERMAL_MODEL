from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd 


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
            for folder in ["Annotated_Frames", "X_Profiles", "Y_Profiles", "Cooling_Histories"]:

                (self.output_dir / f"Pass_{pass_number}" / folder).mkdir(parents=True, exist_ok=True)

        (self.output_dir / "Summary_Figures").mkdir(parents=True, exist_ok=True)

 
    # COOLING HISTORY

    def plot_cooling_history(self, table, pass_number, block_number, output_path = None):

        data = table[
            (table["pass"] == pass_number) & (table["block"] == block_number)].copy()

        data = data.sort_values("cooling_time_s")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["cooling_time_s"], data["center_T_exp_C"], marker="o", label="FLIR")
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
        pass_number,
        block_number,
        cooling_time,
        validation_step,
        output_path=None):

        names = ["x_minus", "center", "x_plus"]

        x_exp = np.array([
            experimental_points[name]["x"]
            for name in names
        ])

        T_exp = np.array([
            experimental_points[name]["temperature"]
            for name in names
        ])

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            x_exp * 1000,
            T_exp,
            marker="o",
            label="FLIR"
        )

        ax.set_xlabel("X Position [mm]")
        ax.set_ylabel("Temperature [°C]")

        ax.set_title(
            f"Pass {pass_number} | Block {block_number} | "
            f"Cooling Time = {cooling_time:.3f} s"
        )

        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if output_path is None:
            output_path = (
                self.output_dir
                / f"Pass_{pass_number}"
                / "X_Profiles"
                / f"Step_{validation_step}_Block_{block_number}.png"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_path, dpi=300)
        plt.close(fig)


    # Y TEMPERATURE PROFILE

    def plot_y_profile(
        self,
        experimental_points,
        pass_number,
        block_number,
        cooling_time,
        validation_step,
        output_path=None):

        names = ["y_minus", "center", "y_plus"]

        y_exp = np.array([
            experimental_points[name]["y"]
            for name in names
        ])

        T_exp = np.array([
            experimental_points[name]["temperature"]
            for name in names
        ])

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            y_exp * 1000,
            T_exp,
            marker="o",
            label="FLIR"
        )

        ax.set_xlabel("Y Position [mm]")
        ax.set_ylabel("Temperature [°C]")

        ax.set_title(
            f"Pass {pass_number} | Block {block_number} | "
            f"Cooling Time = {cooling_time:.3f} s"
        )

        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        if output_path is None:
            output_path = (
                self.output_dir
                / f"Pass_{pass_number}"
                / "Y_Profiles"
                / f"Step_{validation_step}_Block_{block_number}.png"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
        image = ax.imshow(temperature_field, cmap="turbo", origin="lower", extent=[0, self.substrate_length, 0, self.substrate_width ],aspect="equal")


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

                ax.scatter(x, y, s=40, edgecolors="black", linewidths=0.75, zorder=3 )

            ax.text(roi["x_center"], roi["y_max"], f"B{block_number}\n{tracked['cooling_time']:.2f} s", horizontalalignment = "center", verticalalignment = "bottom",  fontsize = 9)


        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        if actual_time is None:
            title = ( f"Pass {pass_number} | Validation Step {validation_step} | FLIR Frame {frame_count}")
               

        else:

            title = (f"Pass {pass_number} | Validation Step {validation_step} t = {actual_time:.3f} s | FLIR Frame {frame_count}")
               
        ax.set_title( title)
        fig.colorbar(image, ax=ax, label="Temperature [°C]")  #Colorbar
        fig.tight_layout()

        # ------ SAVE ----------
        if output_path is None:
            output_path = (
                self.output_dir / f"Pass_{pass_number}" / "Annotated_Frames" / (
                    f"Step_{validation_step}_ Frame_{frame_count}.png"))

        print("Saved annotated frame: {output_path}")

        fig.savefig(output_path, dpi=300 )
        plt.close(fig)



    #Save temperature history 

    def save_temperature_history(self, table):

        point_names = ["center", "x_minus", "x_plus", "y_minus", "y_plus"]
        records = []

        for _, row in table.iterrows():
            for point in point_names:
                records.append({
                    "pass": row["pass"],
                    "validation_step": row["validation_step"],
                    "block": row["block"],
                    "cooling_time_s": row["cooling_time_s"],
                    "point": point,
                    "T_FLIR_C": row[f"{point}_T_exp_C"],
                    "T_model_C": row[f"{point}_T_sim_C"],
                    "error_C": row[f"{point}_error_C"]
                })

        history = pd.DataFrame(records)

        output = self.output_dir / "Tables" / "temperature_history.csv"
        output.parent.mkdir(parents=True, exist_ok=True)

        history.to_csv(output, index=False)

        return history


    def save_summary_statistics(self, table):

        summary = (
            table.groupby(["pass", "block"])
            .agg(
                MAE_C=("MAE_C", "mean"),
                RMSE_C=("RMSE_C", "mean"),
                Max_Error_C=("max_abs_error_C", "max"),
                Observations=("block", "size")
            )
            .reset_index()
        )

        output = self.output_dir / "Tables" / "summary_statistics.csv"
        output.parent.mkdir(parents=True, exist_ok=True)

        summary.to_csv(output, index=False)

        return summary


    def plot_error_by_block(self, summary):

        fig, ax = plt.subplots(figsize=(10, 5))

        for pass_number, data in summary.groupby("pass"):
            ax.plot(
                data["block"],
                data["MAE_C"],
                marker="o",
                label=f"Pass {pass_number}"
            )

        ax.set_xlabel("Deposited Block")
        ax.set_ylabel("Mean Absolute Error (°C)")
        ax.set_title("Model Temperature Error by Block")
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()

        output = (
            self.output_dir
            / "Summary_Figures"
            / "error_by_block.png"
        )

        fig.savefig(output, dpi=300)
        plt.close(fig)


    



if __name__ == "__main__":

    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd

    from spatial_calibration import SpatialCalibration
    from thermal_validation import ThermalValidation

 
    # 1. PROJECT PATHS
    BASE_DIR = Path(__file__).resolve().parent

    PROJECT_DIR = BASE_DIR.parent.parent.parent
    PYTHON_DIR = BASE_DIR.parent.parent
    GRAPH_CODE_DIR = PYTHON_DIR / "GraphCode"
    OUTPUT_DIR = PROJECT_DIR / "Validation" 

    sys.path.insert(0, str(PYTHON_DIR))
    sys.path.insert(0, str(GRAPH_CODE_DIR))

    from GraphCode.config import BUILD, BLOCK, WAAM
    from GraphCode.geometry import Geometry
    from GraphCode.material import WireMaterial, SubstrateMaterial

  
    # 2. BUILD GEOMETRY
    substrate_material = SubstrateMaterial
    wire_material = WireMaterial

    stl_file = BUILD["stl_file"]
    geom = Geometry(stl_file, wire_material, substrate_material)
    geom.build_blocks()


    # 3. EXPERIMENTAL DATA PATHS
    THERMAL_DATA_DIR = (PROJECT_DIR / "A36Steel_Info" / "ThermalData")
    CALIBRATION_FILE = (THERMAL_DATA_DIR / "spatial_calibration.npz")
    DOWNLOADS_DIR = Path.home() / "Downloads"

    VALIDATION_DIR = OUTPUT_DIR

    PROCESSED_DIR = (
         DOWNLOADS_DIR
        / "Thermal_Processed"
        / "Temperature_Data"
    )

    EXPERIMENTAL_DIR = (
        VALIDATION_DIR
        / "Results"
        / "Experimental"
    )

    EXPERIMENTAL_DIR.mkdir(parents=True, exist_ok=True)


    # ============================================================
    # 4. LOAD SPATIAL CALIBRATION
    # ============================================================

    calibration = SpatialCalibration()

    calibration.load_calibration(
        CALIBRATION_FILE
    )

    # ============================================================
    # 5. CREATE VALIDATION OBJECT
    # ============================================================

    validation = ThermalValidation(
        calibration=calibration,
        blocks_per_layer=BUILD["blocks_per_layer"],
        bead_length=geom.length,
        block_width=BLOCK["width"],
        scan_speed=WAAM["scan_speed"]
    )

    # ============================================================
    # 6. CREATE GRAPHING OBJECT
    # ============================================================

    graphs = ThermalValidationGraphs(
        validation=validation,
        substrate_length=BUILD["substrate_length"],
        substrate_width=BUILD["substrate_width"],
        output_dir=EXPERIMENTAL_DIR
    )

    # ============================================================
    # 7. CREATE BLOCK REGIONS OF INTEREST
    # ============================================================

    bead_x_start = BUILD["x_start"]

    bead_y_center = (
        BUILD["substrate_width"] / 2
        + 0.001 / 2
    )

    validation.create_block_rois(
        bead_x_start=bead_x_start,
        bead_y_center=bead_y_center
    )

    # ============================================================
    # 8. INITIALIZE EXPERIMENTAL DATA COLLECTION
    # ============================================================

    experimental_records = []
    cooling_histories = []

    # ============================================================
    # 9. PROCESS EACH EXPERIMENTAL PASS
    # ============================================================

    for pass_number in [1, 2]:

        print("\n======================================")
        print(f"Processing Validation Pass {pass_number}")
        print("======================================")

        # --------------------------------------------------------
        # A. LOAD FLIR TIMING CSV
        # --------------------------------------------------------

        csv_path = (
            THERMAL_DATA_DIR
            / "Processed"
            / f"Pass_{pass_number}"
            / f"A700_Pass{pass_number}.csv"
        )

        timing = validation.load_a700_timing(
            csv_path
        )

        # --------------------------------------------------------
        # B. PROCESSED TEMPERATURE DIRECTORY
        # --------------------------------------------------------

        pass_directory = (
            PROCESSED_DIR / f"Pass_{pass_number}"
        )

        if not pass_directory.exists():
            raise FileNotFoundError(
                f"Temperature directory not found: {pass_directory}"
            )

        # ========================================================
        # C. SELECTED VALIDATION FRAMES
        # ========================================================

        for validation_step in range(
            1,
            validation.blocks_per_layer + 1
        ):

            # Target simulation time
            target_time = (
                validation_step
                * validation.time_per_block
            )

            print(
                f"\nValidation Step {validation_step}"
                f" | Target time = {target_time:.3f} s"
            )

            # ----------------------------------------------------
            # Find nearest experimental frame
            # ----------------------------------------------------

            nearest_frame = validation.find_nearest_frame(
                timing,
                target_time
            )

            frame_count = int(
                nearest_frame["Count"]
            )

            actual_time = float(
                nearest_frame["elapsed_time"]
            )

            print(
                f"Selected frame: {frame_count}"
                f" | Actual time: {actual_time:.3f} s"
                f" | Error: {actual_time - target_time:.4f} s"
            )

            # ----------------------------------------------------
            # Locate processed temperature image
            # ----------------------------------------------------

            pattern = f"{frame_count:08d}_*.npy"

            matching_files = sorted(
                pass_directory.glob(pattern)
            )

            if not matching_files:
                print(
                    f"WARNING: No temperature field for "
                    f"frame {frame_count}"
                )
                continue

            if len(matching_files) > 1:
                print(
                    f"WARNING: Multiple files for frame "
                    f"{frame_count}. Using {matching_files[0]}"
                )

            temperature_field = np.load(
                matching_files[0]
            )

            # ----------------------------------------------------
            # Identify all previously deposited blocks
            # ----------------------------------------------------

            tracked_blocks = (
                validation.get_tracked_blocks(
                    validation_step
                )
            )

            # ----------------------------------------------------
            # Extract experimental measurements
            # ----------------------------------------------------

            for tracked in tracked_blocks:

                block_number = tracked["block"]

                roi = validation.block_rois[
                    block_number - 1
                ]

                experimental_points = (
                    validation.extract_point_temperatures(
                        temperature_field,
                        roi
                    )
                )

                record = (
                    validation.create_experimental_record(
                        pass_number=pass_number,
                        frame_count=frame_count,
                        validation_step=validation_step,
                        block_number=block_number,
                        cooling_time=tracked["cooling_time"],
                        experimental_points=experimental_points,
                        actual_time=actual_time,
                        temperature_field=temperature_field
                    )
                )

                experimental_records.append(record)

            # ----------------------------------------------------
            # Plot annotated experimental frame
            # ----------------------------------------------------

            graphs.plot_validation_frame(
                temperature_field=temperature_field,
                tracked_blocks=tracked_blocks,
                pass_number=pass_number,
                frame_count=frame_count,
                validation_step=validation_step,
                actual_time=actual_time
            )

        # ========================================================
        # D. CONTINUOUS EXPERIMENTAL COOLING HISTORY
        # ========================================================

        print(
            f"\nExtracting continuous cooling history "
            f"for Pass {pass_number}..."
        )

        cooling_history = (
            validation.extract_cooling_history(
                timing=timing,
                pass_directory=pass_directory,
                pass_number=pass_number
            )
        )

        if not cooling_history.empty:

            cooling_histories.append(
                cooling_history
            )

            print(
                f"Cooling observations extracted: "
                f"{len(cooling_history)}"
            )

        else:
            print(
                f"WARNING: No cooling-history data "
                f"for Pass {pass_number}"
            )

    # ============================================================
    # 10. SAVE SELECTED VALIDATION MEASUREMENTS
    # ============================================================

    experimental_table = (
        validation.create_experimental_table(
            experimental_records
        )
    )

    tracking_output = (
        EXPERIMENTAL_DIR
        / "experimental_tracking_history.csv"
    )

    validation.save_experimental_table(
        experimental_table,
        tracking_output
    )

    # ============================================================
    # 11. SAVE CONTINUOUS COOLING HISTORY
    # ============================================================

    if cooling_histories:

        all_cooling_history = pd.concat(
            cooling_histories,
            ignore_index=True
        )

        all_cooling_history = (
            all_cooling_history.sort_values(
                [
                    "pass",
                    "block",
                    "actual_cooling_time_s"
                ]
            ).reset_index(drop=True)
        )

        cooling_output = (
            EXPERIMENTAL_DIR
            / "experimental_cooling_history.csv"
        )

        all_cooling_history.to_csv(
            cooling_output,
            index=False
        )

        print(
            f"\nSaved cooling history: {cooling_output}"
        )

    # ============================================================
    # 12. FINAL SUMMARY
    # ============================================================

    print("\n======================================")
    print("EXPERIMENTAL EXTRACTION COMPLETE")
    print("======================================")

    print(
        f"Selected validation observations: "
        f"{len(experimental_table)}"
    )

    if cooling_histories:
        print(
            f"Continuous cooling observations: "
            f"{len(all_cooling_history)}"
        )

    print(f"Output directory: {EXPERIMENTAL_DIR}")
