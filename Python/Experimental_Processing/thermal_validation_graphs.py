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
        ax.set_title(f"Pass {pass_number} - "Block {block_number} ")
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
            f"{cooling_time:.3f} s"
        )

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

            for name, (x, y) in points.items():

                ax.scatter(x, y, s=35)

            ax.text(roi["x_center"],
                    roi["y_max"],

                (f"B{block_number}\n"
                 f"{tracked['cooling_time']:.2f} s"),

                horizontalalignment="center",
                verticalalignment="bottom"
            )


        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Pass {pass_number} | FLIR Frame {frame_count} ")

        fig.colorbar(image, ax=ax, label="Temperature [°C]")
        fig.tight_layout()

        if output_path is None:

            output_path = (
                self.output_dir
                / f"Pass_{pass_number}"
                / "Annotated_Frames"
                / (f"Step_{validation_step}_"
                    f"Frame_{frame_count}.png")
            )


        fig.savefig(output_path, dpi=300)

        plt.close(fig)