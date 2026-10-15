"""Run the WAAM graph thermal model, save results, and create diagnostics.

Keep calibration in calibrate_graph.py and animations in Visualization/.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from geometry import Geometry
from nodes import NodeGenerator
from graph import ThermalGraph
from material import WireMaterial, SubstrateMaterial
from deposition import DepositionSimulation
from thermocouples import ThermalProbe, middle_block_of_layer, check_probe_location
from config import MATERIAL, GRAPH, CONVECTION, BLOCK, DWELL, BUILD, SENSORS, WAAM

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_DIR / "Validation" / "Results" / "Simulation"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Results directory: {RESULT_DIR.resolve()}")


def create_geometry_and_nodes():
    geometry = Geometry(
        BUILD["stl_file"],
        substrate_material=SubstrateMaterial(),
        wire_material=WireMaterial(),
    )
    geometry.build_blocks()
    generator = NodeGenerator(geometry)
    generator.generate()
    generator.validate_nodes()
    generator.print_surface_statistics()
    return geometry, generator


def create_probes(geometry):
    thermocouples = []
    for name, sensor in SENSORS.items():
        position = np.array([
            sensor["x"], sensor["y"], geometry.substrate_top + sensor["z"]
        ], dtype=float)
        probe = ThermalProbe(name=name, position=position, region=sensor["region"])
        check_probe_location(probe, geometry)
        probe.summary()
        thermocouples.append(probe)

    middle_block = middle_block_of_layer(geometry, layer=0)
    meltpool_probe = ThermalProbe(
        name="meltpool_probe", position=middle_block.center,
        region="deposition", layer=0,
    )
    check_probe_location(meltpool_probe, geometry)
    meltpool_probe.summary()
    return thermocouples, meltpool_probe


def create_graph(nodes):
    graph = ThermalGraph(nodes)
    graph.build()
    graph.degree_matrix()
    graph.laplacian()
    graph.eigensystem()
    graph.summary()
    return graph


def run_simulation(geometry, generator, graph, probes):
    simulation = DepositionSimulation(
        geometry=geometry,
        node_generator=generator,
        graph=graph,
        material=MATERIAL,
        convection=CONVECTION,
        graph_settings=GRAPH,
        dwell_info=DWELL,
        dwell_time=DWELL["dwell"],
        block_info=BLOCK,
        build_info=BUILD,
        probes=probes,
        waam=WAAM,
    )
    simulation.run()
    return simulation


def collect_results(simulation, nodes, thermocouples, meltpool_probe):
    tracker = simulation.tracker
    tc_names = [probe.name for probe in thermocouples]

    result = {
        "temperature": np.asarray(tracker.history, dtype=float),
        "time": np.asarray(tracker.history_time, dtype=float),
        "active_history": np.asarray(tracker.active_history, dtype=bool),
        "node_positions": np.asarray([n.position for n in nodes], dtype=float),
        "substrate_mask": np.asarray([n.is_substrate for n in nodes], dtype=bool),
        "node_block_ids": np.asarray([
            n.block_id if n.block_id is not None else -1 for n in nodes
        ], dtype=int),
        "probe_time": np.asarray(tracker.probe_history_time, dtype=float),
        "tc_names": np.asarray(tc_names),
        "tc_temperature": np.column_stack([
            np.asarray(tracker.probe_history[name], dtype=float) for name in tc_names
        ]),
        "tc_positions": np.asarray([p.position for p in thermocouples], dtype=float),
        "meltpool_temperature": np.asarray(
            tracker.probe_history[meltpool_probe.name], dtype=float
        ),
        "meltpool_position": np.asarray(meltpool_probe.position, dtype=float),
    }

    # The current tracker uses 'laser' terminology for the moving WAAM heat source.
    if hasattr(tracker, "laser_position_history"):
        result["laser_position_history"] = np.asarray(
            tracker.laser_position_history, dtype=float
        )
    elif hasattr(tracker, "heat_source_position_history"):
        result["laser_position_history"] = np.asarray(
            tracker.heat_source_position_history, dtype=float
        )
    else:
        # No position history recorded; NaN tells visualization to hide the source.
        result["laser_position_history"] = np.full((len(result["time"]), 3), np.nan)
        print("WARNING: No heat-source position history found in tracker.")

    n_frames, n_nodes = result["temperature"].shape
    if result["time"].shape != (n_frames,):
        raise ValueError("History times and temperatures do not align")
    if result["active_history"].shape != (n_frames, n_nodes):
        raise ValueError("Active history does not match temperature history")
    if result["node_positions"].shape != (n_nodes, 3):
        raise ValueError("Node positions do not match temperature history")
    if result["laser_position_history"].shape != (n_frames, 3):
        raise ValueError("Heat-source positions must have shape (n_frames, 3)")
    if result["tc_temperature"].shape != (len(result["probe_time"]), len(tc_names)):
        raise ValueError("Thermocouple temperatures and probe times do not align")
    if result["meltpool_temperature"].shape != result["probe_time"].shape:
        raise ValueError("Meltpool temperatures and probe times do not align")
    return result

def plot_experimental_thermocouples():
    """
    Plot experimental thermocouple temperatures
    from the processed Pass 1 CSV.
    """

    # GraphThermalModel/
    PROJECT_DIR = Path(__file__).resolve().parents[2]

    csv_file = (
        PROJECT_DIR
        / "A36Steel_Info"
        / "ThermalData"
        / "Processed"
        / "Pass_1"
        / "Thermocouples_Pass1.csv"
    )

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Experimental thermocouple data not found:\n{csv_file}"
        )

    df = pd.read_csv(csv_file)

    # Convert time to numeric
    df["RelativeTime_s"] = pd.to_numeric(
        df["RelativeTime_s"],
        errors="coerce"
    )

    # Keep deposition start and later
    df = df[df["RelativeTime_s"] >= 0].copy()

    tc_columns = ["ch0", "ch1", "ch2", "ch3"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, col in enumerate(tc_columns):

        # Convert invalid values such as FALSE to NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep valid measurements
        valid = df[["RelativeTime_s", col]].dropna()

        # Sort by time
        valid = valid.sort_values("RelativeTime_s")

        ax.plot(
            valid["RelativeTime_s"],
            valid[col],
            linewidth=1.5,
            label=f"TC_{i} (Experimental)"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Experimental Thermocouple Temperature Histories")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    save_figure(fig, "experimental_thermocouple_history.png")

    print(f"Loaded experimental data: {csv_file}")

    return fig


def plot_experimental_vs_simulated(data):
    """
    Compare experimental and simulated thermocouple
    temperature histories on a single plot.
    """

    PROJECT_DIR = Path(__file__).resolve().parents[2]

    csv_file = (
        PROJECT_DIR
        / "A36Steel_Info"
        / "ThermalData"
        / "Processed"
        / "Pass_1"
        / "Thermocouples_Pass1.csv"
    )

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Experimental data not found: {csv_file}"
        )

    experimental = pd.read_csv(csv_file)

    # Experimental time
    experimental["RelativeTime_s"] = pd.to_numeric(
        experimental["RelativeTime_s"],
        errors="coerce"
    )

    experimental = experimental[
        experimental["RelativeTime_s"] >= 0
    ].copy()

    # Mapping between experimental channels and simulation probes
    tc_mapping = {
        "TC_0": "ch0",
        "TC_1": "ch1",
        "TC_2": "ch2",
        "TC_3": "ch3"
    }

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (tc_name, channel) in enumerate(tc_mapping.items()):

        color = colors[i]

        # -----------------------------
        # Experimental temperatures
        # -----------------------------
        experimental[channel] = pd.to_numeric(
            experimental[channel],
            errors="coerce"
        )

        valid = experimental[
            ["RelativeTime_s", channel]
        ].dropna().sort_values("RelativeTime_s")

        ax.plot(
            valid["RelativeTime_s"],
            valid[channel],
            color=color,
            linestyle="-",
            linewidth=1.8,
            label=f"{tc_name} Experimental"
        )

        # -----------------------------
        # Simulated temperatures
        # -----------------------------
        if tc_name not in data["tc_names"]:
            raise ValueError(
                f"{tc_name} not found in simulation thermocouples."
            )

        sim_index = list(data["tc_names"]).index(tc_name)

        ax.plot(
            data["probe_time"],
            data["tc_temperature"][:, sim_index],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"{tc_name} Simulated"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Experimental vs. Simulated Thermocouple Temperatures")

    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()

    save_figure(fig, "experimental_vs_simulated_thermocouples.png")

    return fig


def save_results(data):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "thermal_history.npz"
    np.savez_compressed(path, **data)
    print(f"Saved simulation data: {path}")
    return path


def save_figure(fig, filename):
    path = RESULT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close(fig)


def plot_line_histories(data):
    t = data["probe_time"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(data["tc_names"]):
        ax.plot(t, data["tc_temperature"][:, i], linewidth=1.7, label=name)
    ax.set(xlabel="Time (s)", ylabel="Temperature (°C)",
           title="Substrate Thermocouple Temperature Histories")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "thermocouple_history.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, data["meltpool_temperature"], label="Meltpool probe")
    ax.set(xlabel="Time (s)", ylabel="Temperature (°C)",
           title="Deposition Probe Temperature History")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "meltpool_probe_history.png")

    # Exclude inactive nodes from global statistics.
    active_temperatures = np.where(data["active_history"], data["temperature"], np.nan)
    valid_rows = np.any(np.isfinite(active_temperatures), axis=1)
    global_max = np.full(len(data["time"]), np.nan)
    global_mean = np.full(len(data["time"]), np.nan)
    if np.any(valid_rows):
        global_max[valid_rows] = np.nanmax(active_temperatures[valid_rows], axis=1)
        global_mean[valid_rows] = np.nanmean(active_temperatures[valid_rows], axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["time"], global_max, label="Maximum active-node temperature")
    ax.plot(data["time"], global_mean, label="Mean active-node temperature")
    ax.set(xlabel="Time (s)", ylabel="Temperature (°C)",
           title="Global Thermal Evolution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "global_temperature_history.png")

    tc_max = [np.nanmax(col) if np.any(np.isfinite(col)) else np.nan
              for col in data["tc_temperature"].T]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(data["tc_names"], tc_max)
    ax.set(xlabel="Thermocouple", ylabel="Maximum Temperature (°C)",
           title="Maximum Temperature at Each Thermocouple")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "thermocouple_maximum.png")


def plot_final_temperature(data):
    mask = data["active_history"][-1]
    xyz = data["node_positions"][mask] * 1000
    temp = data["temperature"][-1, mask]
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=temp,
                         cmap="inferno", s=6)
    fig.colorbar(scatter, ax=ax, shrink=0.65, label="Temperature (°C)")
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
           title="Final 3D Temperature Distribution")
    fig.tight_layout()
    save_figure(fig, "final_3d_temperature.png")


def plot_probe_locations(data):
    xyz = data["node_positions"] * 1000
    substrate = data["substrate_mask"]
    fig = plt.figure(figsize=(15, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    for mask, color, name in [
        (substrate, "gray", "Substrate"),
        (~substrate, "orange", "Deposition"),
    ]:
        points = xyz[mask]
        if len(points):
            ax3d.scatter(*points.T, s=3, alpha=0.15, color=color, label=name)
            ax2d.scatter(points[:, 0], points[:, 1], s=3, alpha=0.15, color=color)

    for name, point in zip(data["tc_names"], data["tc_positions"] * 1000):
        x, y, z = point
        ax3d.scatter(x, y, z, s=90, edgecolor="black", label=str(name))
        ax3d.text(x, y, z + 1, str(name))
        ax2d.scatter(x, y, s=90, edgecolor="black")
        ax2d.annotate(str(name), (x, y), xytext=(5, 5), textcoords="offset points")

    x, y, z = data["meltpool_position"] * 1000
    ax3d.scatter(x, y, z, s=130, marker="*", color="red", label="Meltpool probe")
    ax2d.scatter(x, y, s=130, marker="*", color="red")
    ax2d.annotate("Meltpool probe", (x, y), xytext=(5, 8), textcoords="offset points")
    ax3d.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
             title="3D Thermocouple Locations")
    ax3d.legend(fontsize=8)
    ax2d.set(xlabel="X (mm)", ylabel="Y (mm)", title="Top View: Thermocouple Locations")
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "thermocouple_locations.png")


def print_summary(data):
    print("\n========== SIMULATION SUMMARY ==========")
    print("Maximum global temperature:", np.nanmax(data["temperature"]))
    print("Stored states:", len(data["time"]))
    print("Final simulated time:", data["time"][-1], "s")
    print("\n========== THERMOCOUPLE SUMMARY ==========")
    for name, values in zip(data["tc_names"], data["tc_temperature"].T):
        valid = values[np.isfinite(values)]
        if len(valid):
            print(f"{name}: min={valid.min():.2f} °C, "
                  f"max={valid.max():.2f} °C, final={valid[-1]:.2f} °C")
        else:
            print(f"{name}: no valid readings")


def print_diagnostics(data, graph):
    xyz = data["node_positions"]
    substrate = data["substrate_mask"]
    far_end = (~substrate) & (xyz[:, 0] > 130e-3)
    print("\n------ FAR-END DEPOSITION CHECK ------")
    print("Number of far-end nodes:", np.count_nonzero(far_end))
    if np.any(far_end):
        final_active_far_end = far_end & data["active_history"][-1]
        if np.any(final_active_far_end):
            print("Final far-end maximum temperature:",
                  np.nanmax(data["temperature"][-1, final_active_far_end]))
        far_end_history = np.where(data["active_history"][:, far_end],
                                   data["temperature"][:, far_end], np.nan)
        if np.any(np.isfinite(far_end_history)):
            print("Peak far-end temperature:", np.nanmax(far_end_history))

    degrees = np.asarray(graph.A.sum(axis=1)).ravel()
    print("\n------ GRAPH CONNECTIVITY ------")
    for name, mask in [
        ("Left", substrate & (xyz[:, 0] < 10e-3)),
        ("Right", substrate & (xyz[:, 0] > 142e-3)),
    ]:
        if np.any(mask):
            print(f"{name} mean weighted degree: {degrees[mask].mean():.6f}")

    substrate_xyz = xyz[substrate]
    print("\n------ PROBE DISTANCES ------")
    if len(substrate_xyz):
        for name, position in zip(data["tc_names"], data["tc_positions"]):
            distance_mm = np.linalg.norm(substrate_xyz - position, axis=1).min() * 1000
            print(f"{name} nearest substrate node: {distance_mm:.3f} mm")



def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    geometry, generator = create_geometry_and_nodes()
    thermocouples, meltpool_probe = create_probes(geometry)
    graph = create_graph(generator.nodes)
    simulation = run_simulation(
        geometry, generator, graph, thermocouples + [meltpool_probe]
    )
    data = collect_results(simulation, generator.nodes, thermocouples, meltpool_probe)
    print_summary(data)
    print_diagnostics(data, graph)
    save_results(data)
    plot_line_histories(data)
    plot_final_temperature(data)
    plot_probe_locations(data)
    plot_experimental_thermocouples()
    plot_experimental_vs_simulated(data)
    print("\nSimulation and plotting complete.")


if __name__ == "__main__":
    main()
