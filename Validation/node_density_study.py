#Compares simulated temperature value to experimental thermocouple data 

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

#-----------------------------
#FILE PATHS

VALIDATION_DIR = Path(__file__).resolve().parent

PYTHON_DIR = VALIDATION_DIR.parent

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_DIR)
    )

RESULTS_DIR = (
    PYTHON_DIR
    / "Results"
    / "Temperature Convergence")

DATA_DIR = RESULTS_DIR / "data"
FIGURE_DIR = RESULTS_DIR / "figures"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True)

from geometry import Geometry
from nodes import NodeGenerator
from graph import ThermalGraph
from deposition import DepositionSimulation
from thermocouples import ThermalProbe

from config import (
    MATERIAL,
    LASER,
    GRAPH,
    CONVECTION,
    BLOCK,
    DWELL,
    BUILD, 
    SENSORS)

# ---------------------------------
# EXPERIMENTAL CASE-A DATA

# Temporary approximate points from paper
# Replace later with full digitized experimental curve

time_exp = np.array([250.0, 1250.0])
T_exp = np.array([200.0, 150.0])


# ---------------------------------
# CONVERGENCE CASES

cases = [

    #case 1
    {"name": "Density 0.2355",
    "node_density": 0.2355,
    "epsilon": 4.50e-3,
    "gain": 10.0,
    "meltpool_temperature": 2200},

    #case 2
    {"name": "Density 0.4709",
    "node_density": 0.4709,
    "epsilon": 4.75e-3,
    "gain": 1.5,
    "meltpool_temperature": 2200},

    #case 3
    {"name": "Density 0.7064",
    "node_density": 0.7064,
    "epsilon": 5.50e-3,
    "gain": 0.15,
    "meltpool_temperature": 2200}
]


# ---------------------------------
# ERROR METRICS

def calculate_errors(
        simulation_time,
        simulation_temperature,
        experimental_time,
        experimental_temperature
    ):

        simulation_time = np.asarray(
            simulation_time,
            dtype=float
        )

        simulation_temperature = np.asarray(
            simulation_temperature,
            dtype=float
        )

        experimental_time = np.asarray(
            experimental_time,
            dtype=float
        )

        experimental_temperature = np.asarray(
            experimental_temperature,
            dtype=float
        )


        # --------------------------------------------------------
        # Check simulation coverage
        # --------------------------------------------------------

        if experimental_time.min() < simulation_time.min():

            raise ValueError(
                "Experimental data begin before "
                "the simulated time range."
            )

        if experimental_time.max() > simulation_time.max():

            raise ValueError(
                "\nExperimental comparison cannot be performed.\n"
                f"Latest experimental time = "
                f"{experimental_time.max():.2f} s\n"
                f"Simulation ends at = "
                f"{simulation_time.max():.2f} s\n\n"
                "np.interp would otherwise silently use the "
                "last simulated temperature for the missing times."
            )


        # --------------------------------------------------------
        # Interpolate model onto experimental timestamps
        # --------------------------------------------------------

        T_sim = np.interp(
            experimental_time,
            simulation_time,
            simulation_temperature
        )


        # --------------------------------------------------------
        # RMSE
        # --------------------------------------------------------

        rmse = np.sqrt(
            np.mean(
                (
                    T_sim
                    - experimental_temperature
                ) ** 2
            )
        )


        # --------------------------------------------------------
        # MAPE
        # --------------------------------------------------------

        if np.any(
            experimental_temperature == 0
        ):

            mape = np.nan

        else:

            mape = (
                np.mean(
                    np.abs(
                        (
                            T_sim
                            - experimental_temperature
                        )
                        / experimental_temperature
                    )
                )
                * 100
            )

        return (
            T_sim,
            mape,
            rmse
        )

#-------------------------------------------
# NUMERICAL CONVERGENCE METRICS


def calculate_convergence_error(
    reference_time,
    reference_temperature,
    comparison_time,
    comparison_temperature):

    # Restrict comparison to common simulated time interval
    common_end_time = min(
        reference_time[-1],
        comparison_time[-1])

    mask = (reference_time <= common_end_time)

    t_common = reference_time[mask]

    T_reference = (reference_temperature[mask])

    # Interpolate comparison onto reference time grid
    T_comparison = np.interp(t_common, comparison_time,comparison_temperature)

    difference = (T_comparison - T_reference)

    # RMSE
    rmse = np.sqrt(np.mean(difference ** 2))

    # Maximum absolute temperature difference
    max_abs_error = np.max(np.abs(difference))

    # Relative L2 norm
    denominator = np.linalg.norm(T_reference)

    if denominator > 0:

        relative_l2 = (np.linalg.norm(difference) / denominator* 100)

    else:

        relative_l2 = np.nan

    return (
        rmse,
        max_abs_error,
        relative_l2,
        t_common,
        T_comparison,
        T_reference)

#----------------------------------
def percent_difference(
    value,
    reference):

    if reference == 0:
        return np.nan

    return (
        abs(value - reference)
        / abs(reference)
        * 100
    )

# ----------------------------------------------
# RUN ONE CONVERGENCE CASE

def run_case(case):

    print("\n" + "=" * 70)
    print(
        f"RUNNING: {case['name']}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Case parameters
    # --------------------------------------------------------

    GRAPH["node_density"] = (
        case["node_density"]
    )

    GRAPH["epsilon"] = (
        case["epsilon"]
    )

    GRAPH["gain"] = (
        case["gain"]
    )

    LASER["meltpool_temperature"] = (
        case["meltpool_temperature"]
    )


    # --------------------------------------------------------
    # Total computation timer
    # --------------------------------------------------------

    compute_start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    geometry = Geometry(
        BUILD["stl_file"]
    )

    geometry.build_blocks()


    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    generator = NodeGenerator(
        geometry
    )

    generator.generate()

    nodes = generator.nodes


    # --------------------------------------------------------
    # Passive TC-A
    # --------------------------------------------------------

    tc_a_config = (
        SENSORS["TC_A"]
    )

    tc_a_position = np.array([
        tc_a_config["x"],
        tc_a_config["y"],
        geometry.substrate_top
        + tc_a_config["z"]
    ])

    TC_A = ThermalProbe(
        name="TC_A",
        position=tc_a_position,
        region="substrate"
    )

    print(
        "\nTC-A position [mm]:",
        TC_A.position * 1000
    )


    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    graph = ThermalGraph(
        nodes
    )

    graph.build()
    graph.degree_matrix()
    graph.laplacian()
    graph.eigensystem()


    # --------------------------------------------------------
    # Deposition simulation
    # --------------------------------------------------------

    simulation = DepositionSimulation(
        geometry=geometry,
        node_generator=generator,
        graph=graph,
        material=MATERIAL,
        laser=LASER,
        convection=CONVECTION,
        graph_settings=GRAPH,
        dwell_info=DWELL,
        dwell_time=DWELL["case_A"],
        block_info=BLOCK,
        probes=[TC_A]
    )

    simulation.run()


    # --------------------------------------------------------
    # Computation time
    # --------------------------------------------------------

    computation_time = (
        time.perf_counter()
        - compute_start
    )


    # --------------------------------------------------------
    # TC-A history sampled from stored graph states
    # --------------------------------------------------------

    simulation_time = np.asarray(
        simulation.history_time,
        dtype=float
    )

    sensor_temperature = np.array([

        TC_A.interpolate_temperature(
            nodes=nodes,
            temperature=T_state,
            k=8,
            power=2,
            active_only=True
        )

        for T_state
        in simulation.history

    ], dtype=float)


    # --------------------------------------------------------
    # Experimental error
    # --------------------------------------------------------

    experimental_comparison_available = (
        simulation_time[-1]
        >= np.max(time_exp)
    )

    if experimental_comparison_available:

        (
            T_sim_exp,
            mape,
            rmse
        ) = calculate_errors(
            simulation_time,
            sensor_temperature,
            time_exp,
            T_exp
        )

    else:

        T_sim_exp = np.full(
            len(time_exp),
            np.nan
        )

        mape = np.nan
        rmse = np.nan

        print(
            "\nWARNING: Experimental comparison "
            "not available for this run."
        )

        print(
            f"Simulation ends at "
            f"{simulation_time[-1]:.2f} s."
        )

        print(
            f"Experimental data extend to "
            f"{np.max(time_exp):.2f} s."
        )


    # --------------------------------------------------------
    # Node statistics
    # --------------------------------------------------------

    total_nodes = len(
        nodes
    )

    substrate_nodes = sum(
        node.is_substrate
        for node in nodes
    )

    deposition_nodes = (
        total_nodes
        - substrate_nodes
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {

        "Case":
            case["name"],

        "Node Density (nodes/mm^3)":
            case["node_density"],

        "Epsilon (mm)":
            case["epsilon"] * 1000,

        "Gain":
            case["gain"],

        "Meltpool Temperature (C)":
            case["meltpool_temperature"],

        "Total Nodes":
            total_nodes,

        "Substrate Nodes":
            substrate_nodes,

        "Deposition Nodes":
            deposition_nodes,

        "Build Time (s)":
            simulation.time,

        "Computation Time (s)":
            computation_time,

        "Peak TC Temperature (C)":
            np.nanmax(
                sensor_temperature
            ),

        "Final TC Temperature (C)":
            sensor_temperature[-1],

        "MAPE (%)":
            mape,

        "RMSE (C)":
            rmse
    }


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n------ CASE RESULT ------"
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


    if experimental_comparison_available:

        print(
            "\nExperimental point comparison:"
        )

        for (
            t,
            T_reference,
            T_model
        ) in zip(
            time_exp,
            T_exp,
            T_sim_exp
        ):

            print(
                f"t={t:.1f} s | "
                f"Experiment={T_reference:.2f} C | "
                f"Simulation={T_model:.2f} C"
            )


    return (
        result,
        simulation_time,
        sensor_temperature
    )

# ============================================================
# RUN CONVERGENCE STUDY
# ============================================================

if __name__ == "__main__":

    results = []

    thermal_histories = {}

    for case in cases:

        (
            result,
            simulation_time,
            sensor_temperature
        ) = run_case(case)

        results.append(
            result
        )

        thermal_histories[
            case["name"]
        ] = (
            simulation_time,
            sensor_temperature
        )

    # ========================================================
    # NUMERICAL CONVERGENCE ANALYSIS
    # Highest-density solution is used as reference
    # ========================================================

    fine_name = "Density 0.7064"
    medium_name = "Density 0.4709"
    coarse_name = "Density 0.2355"


    fine_time, fine_T = (
        thermal_histories[fine_name]
    )

    medium_time, medium_T = (
        thermal_histories[medium_name]
    )

    coarse_time, coarse_T = (
        thermal_histories[coarse_name]
    )


    # --------------------------------------------------------
    # Coarse vs fine
    # --------------------------------------------------------

    (
        coarse_rmse,
        coarse_max_error,
        coarse_relative_l2,
        coarse_common_time,
        coarse_interp,
        coarse_reference
    ) = calculate_convergence_error(

        reference_time=fine_time,
        reference_temperature=fine_T,

        comparison_time=coarse_time,
        comparison_temperature=coarse_T
    )


    # --------------------------------------------------------
    # Medium vs fine
    # --------------------------------------------------------

    (
        medium_rmse,
        medium_max_error,
        medium_relative_l2,
        medium_common_time,
        medium_interp,
        medium_reference
    ) = calculate_convergence_error(

        reference_time=fine_time,
        reference_temperature=fine_T,

        comparison_time=medium_time,
        comparison_temperature=medium_T
    )


    fine_peak = np.max(fine_T)

    medium_peak = np.max(medium_T)

    coarse_peak = np.max(coarse_T)


    coarse_peak_difference = (
        percent_difference(
            coarse_peak,
            fine_peak
        )
    )

    medium_peak_difference = (
        percent_difference(
            medium_peak,
            fine_peak
        )
    )

    convergence_results = pd.DataFrame({

        "Comparison": [
            "0.2355 vs 0.7064",
            "0.4709 vs 0.7064"
        ],

        "RMSE vs Fine (C)": [
            coarse_rmse,
            medium_rmse
        ],

        "Maximum Difference (C)": [
            coarse_max_error,
            medium_max_error
        ],

        "Relative L2 Error (%)": [
            coarse_relative_l2,
            medium_relative_l2
        ],

        "Peak Temperature Difference (%)": [
            coarse_peak_difference,
            medium_peak_difference
        ]
    })

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NUMERICAL CONVERGENCE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        convergence_results.to_string(
            index=False
        )
    )

    convergence_results.to_csv(
        DATA_DIR
        / "numerical_convergence_results.csv",
        index=False
    )

  

    # ========================================================
    # PLOT 5
    # TEMPERATURE ERROR RELATIVE TO FINE SOLUTION
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )


    # Coarse error
    coarse_error_history = (
        coarse_interp
        - coarse_reference
    )

    plt.plot(
        coarse_common_time,
        np.abs(
            coarse_error_history
        ),
        label="0.2355 vs 0.7064"
    )


    # Medium error
    medium_error_history = (
        medium_interp
        - medium_reference
    )

    plt.plot(
        medium_common_time,
        np.abs(
            medium_error_history
        ),
        label="0.4709 vs 0.7064"
    )


    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Absolute Temperature Difference (°C)"
    )

    plt.title(
        "Temperature-History Convergence Relative to Fine Graph"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "node_density_solution_difference.png",
        dpi=300
    )

    plt.show()

    # ========================================================
    # PLOT 6
    # RELATIVE ERROR VS NODE DENSITY
    # ========================================================

    density_values = np.array([
        0.2355,
        0.4709,
        0.7064
    ])

    relative_errors = np.array([
        coarse_relative_l2,
        medium_relative_l2,
        0.0
    ])


    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        density_values,
        relative_errors,
        marker="o"
    )

    plt.xlabel(
        "Node Density (nodes/mm³)"
    )

    plt.ylabel(
        "Relative L2 Difference from Fine Solution (%)"
    )

    plt.title(
        "Graph Solution Convergence"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "node_density_convergence_error.png",
        dpi=300
    )

    plt.show()

    (
        coarse_medium_rmse,
        coarse_medium_max,
        coarse_medium_l2,
        _,
        _,
        _
    ) = calculate_convergence_error(

        reference_time=medium_time,
        reference_temperature=medium_T,

        comparison_time=coarse_time,
        comparison_temperature=coarse_T
    )

    print(
    "\n------ CONSECUTIVE RESOLUTION CHECK ------"
    )

    print(
        f"Coarse -> Medium RMSE: "
        f"{coarse_medium_rmse:.3f} °C"
    )

    print(
        f"Medium -> Fine RMSE: "
        f"{medium_rmse:.3f} °C"
    )

    if medium_rmse < coarse_medium_rmse:

        print(
            "PASS: solution difference decreases "
            "with increasing graph resolution."
        )

    else:

        print(
            "WARNING: monotonic convergence was "
            "not observed."
        )

    convergence_ratio = (
        medium_rmse
        / coarse_medium_rmse
    )

    print(
        f"Convergence ratio: "
        f"{convergence_ratio:.4f}"
    )
        

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "NODE DENSITY CONVERGENCE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        DATA_DIR
        / "node_density_convergence.csv",
        index=False
    )



    # ========================================================
    # PLOT 1
    # THERMOCOUPLE HISTORY
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )

    for case_name, (
        sim_time,
        sensor_T
    ) in thermal_histories.items():

        plt.plot(
            sim_time,
            sensor_T,
            label=case_name
        )

    plt.scatter(
        time_exp,
        T_exp,
        marker="o",
        s=80,
        label="Experimental"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Temperature (°C)"
    )

    plt.title(
        "Case A Node-Density Convergence Study"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "node_density_temperature_history.png",
        dpi=300
    )

    plt.show()


    # ========================================================
    # PLOT 2
    # MAPE VS NODE DENSITY
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        results_df[
            "Node Density (nodes/mm^3)"
        ],
        results_df[
            "MAPE (%)"
        ],
        marker="o"
    )

    plt.xlabel(
        "Node Density (nodes/mm³)"
    )

    plt.ylabel(
        "MAPE (%)"
    )

    plt.title(
        "MAPE vs Node Density"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "node_density_mape.png",
        dpi=300
    )

    plt.show()


    # ========================================================
    # PLOT 3
    # RMSE VS NODE DENSITY
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        results_df[
            "Node Density (nodes/mm^3)"
        ],
        results_df[
            "RMSE (C)"
        ],
        marker="o"
    )

    plt.xlabel(
        "Node Density (nodes/mm³)"
    )

    plt.ylabel(
        "RMSE (°C)"
    )

    plt.title(
        "RMSE vs Node Density"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "node_density_rmse.png",
        dpi=300
    )

    plt.show()


    # ========================================================
    # PLOT 4
    # COMPUTATION TIME VS NODE DENSITY
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        results_df[
            "Node Density (nodes/mm^3)"
        ],
        results_df[
            "Computation Time (s)"
        ],
        marker="o"
    )

    plt.xlabel(
        "Node Density (nodes/mm³)"
    )

    plt.ylabel(
        "Computation Time (s)"
    )

    plt.title(
        "Computation Time vs Node Density"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "node_density_computation_time.png",
        dpi=300
    )

    plt.show()