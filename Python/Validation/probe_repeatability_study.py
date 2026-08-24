"""
Artificial melt-pool probe repeatability and
graph-resolution convergence study.

The probe is a passive numerical measurement location.
It does NOT create or move a graph node.

Three randomized node realizations are run for each
graph resolution.

Goals:
    1. Quantify sensitivity to random node placement.
    2. Determine whether higher node density reduces
       temperature scatter.
    3. Compare mean probe temperature histories between
       graph resolutions.
    4. Compare computational cost.
"""

# ============================================================
# IMPORTS / PROJECT PATHS
# ============================================================

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Python/Validation/
VALIDATION_DIR = Path(__file__).resolve().parent

# Python/
PYTHON_DIR = VALIDATION_DIR.parent


# Allow Python/ modules to be imported
if str(PYTHON_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(PYTHON_DIR)
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from geometry import Geometry
from nodes import NodeGenerator
from graph import ThermalGraph
from deposition import DepositionSimulation

from thermocouples import (
    ThermalProbe,
    middle_block_of_layer,
    check_probe_location
)

from config import (
    MATERIAL,
    LASER,
    GRAPH,
    CONVECTION,
    BLOCK,
    DWELL,
    BUILD
)


# ============================================================
# RESULTS PATHS
# ============================================================

RESULTS_DIR = (
    VALIDATION_DIR
    / "Results"
    / "MeshConvergence"
)

DATA_DIR = (
    RESULTS_DIR
    / "Data"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "Figures"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GRAPH RESOLUTION CASES
# ============================================================

CASES = [

    {
        "name": "Density 0.2355",
        "node_density": 0.2355,
        "epsilon": 4.50e-3,
        "gain": 10.0,
    },

    {
        "name": "Density 0.4709",
        "node_density": 0.4709,
        "epsilon": 4.75e-3,
        "gain": 1.5,
    },

    {
        "name": "Density 0.7064",
        "node_density": 0.7064,
        "epsilon": 5.50e-3,
        "gain": 0.15,
    },
]


N_REPEATS = 3


# ============================================================
# HELPER
# ============================================================

def density_string(
    density
):

    return (
        str(density)
        .replace(".", "p")
    )


# ============================================================
# RUN ONE RANDOMIZED REALIZATION
# ============================================================

def run_single_case(
    case,
    repeat
):

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"RUNNING {case['name']} "
        f"- Repeat {repeat}/{N_REPEATS}"
    )

    print(
        "=" * 75
    )


    # --------------------------------------------------------
    # Graph parameters
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


    # --------------------------------------------------------
    # Total wall-clock timer
    # --------------------------------------------------------

    total_start = (
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
    # Random nodes
    #
    # No seed is used intentionally.
    # Each repeat receives a new point cloud.
    # --------------------------------------------------------

    generator = NodeGenerator(
        geometry
    )

    generator.generate()

    nodes = (
        generator.nodes
    )


    # --------------------------------------------------------
    # Artificial melt-pool probe
    #
    # Fixed physical position:
    # center of middle block of first deposition layer.
    #
    # Probe is passive.
    # It DOES NOT modify the graph.
    # --------------------------------------------------------

    middle_block = (
        middle_block_of_layer(
            geometry,
            layer=0
        )
    )

    meltpool_probe = ThermalProbe(
        name="meltpool_probe",
        position=middle_block.center.copy(),
        region="deposition",
        layer=0
    )


    print(
        "\n------ ARTIFICIAL PROBE ------"
    )

    print(
        "Position [mm]:",
        meltpool_probe.position
        * 1000
    )

    print(
        "Target block:",
        middle_block.id
    )

    print(
        "Layer:",
        middle_block.layer
    )

    check_probe_location(
        meltpool_probe,
        geometry
    )


    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    graph = ThermalGraph(
        nodes
    )


    graph_start = (
        time.perf_counter()
    )


    graph.build()

    graph.degree_matrix()

    graph.laplacian()

    graph.eigensystem()


    graph_time = (
        time.perf_counter()
        - graph_start
    )


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

        probes=[
            meltpool_probe
        ]
    )


    simulation_start = (
        time.perf_counter()
    )


    simulation.run()


    simulation_computation_time = (
        time.perf_counter()
        - simulation_start
    )


    total_computation_time = (
        time.perf_counter()
        - total_start
    )


    # --------------------------------------------------------
    # Probe history
    #
    # This MUST come from probe_history, not sensor_history(),
    # because the probe is not a graph node.
    # --------------------------------------------------------

    probe_temperature = np.asarray(
        simulation.probe_history[
            "meltpool_probe"
        ],
        dtype=float
    )

    probe_time = np.asarray(
        simulation.probe_history_time,
        dtype=float
    )

    probe_stage = np.asarray(
        simulation.probe_history_stage,
        dtype=str
    )


    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not (
        len(probe_temperature)
        == len(probe_time)
        == len(probe_stage)
    ):

        raise ValueError(
            "Probe history, time, and stage arrays "
            "do not have matching lengths."
        )


    valid = np.isfinite(
        probe_temperature
    )


    if not np.any(valid):

        raise ValueError(
            "Melt-pool probe never returned a finite "
            "temperature during this simulation."
        )


    # --------------------------------------------------------
    # Probe statistics
    # --------------------------------------------------------

    valid_temperature = (
        probe_temperature[
            valid
        ]
    )

    peak_temperature = (
        np.max(
            valid_temperature
        )
    )


    # --------------------------------------------------------
    # Save raw probe history for this realization
    # --------------------------------------------------------

    d_string = density_string(
        case["node_density"]
    )


    run_df = pd.DataFrame({

        "Time (s)":
            probe_time,

        "Stage":
            probe_stage,

        "Probe Temperature (C)":
            probe_temperature
    })


    run_df.to_csv(

        DATA_DIR
        / (
            f"density_{d_string}"
            f"_repeat_{repeat}.csv"
        ),

        index=False
    )


    # --------------------------------------------------------
    # Return realization data
    # --------------------------------------------------------

    return {

        "case":
            case["name"],

        "density":
            case["node_density"],

        "epsilon":
            case["epsilon"],

        "gain":
            case["gain"],

        "repeat":
            repeat,

        "probe_position":
            meltpool_probe.position.copy(),

        "time":
            probe_time.copy(),

        "stage":
            probe_stage.copy(),

        "temperature":
            probe_temperature.copy(),

        "peak_temperature":
            peak_temperature,

        "graph_time":
            graph_time,

        "simulation_computation_time":
            simulation_computation_time,

        "total_computation_time":
            total_computation_time,

        "total_nodes":
            len(nodes)
    }


# ============================================================
# EXTRACT ONE CONSISTENT MEASUREMENT STAGE
# ============================================================

def extract_stage_history(
    run,
    stage_name
):
    """
    Extract one consistent probe-measurement stage.

    Using one stage prevents values such as after-Goldak
    and after-conduction from being compared as though they
    were the same physical state.
    """

    mask = (
        run["stage"]
        == stage_name
    )

    return (
        run["time"][mask],
        run["temperature"][mask]
    )


# ============================================================
# MAIN STUDY
# ============================================================

def main():

    all_runs = []

    summary_results = []


    # ========================================================
    # RUN ALL NINE SIMULATIONS
    # ========================================================

    for case in CASES:

        for repeat in range(
            1,
            N_REPEATS + 1
        ):

            run = run_single_case(
                case,
                repeat
            )

            all_runs.append(
                run
            )


    # ========================================================
    # CHOOSE COMPARISON STATE
    # ========================================================

    # For the artificial melt-pool test, this is useful
    # because it evaluates the probe immediately after
    # application of the Goldak heat source.
    #
    # Change to "after_convection" later if you want the
    # completed block thermal state instead.

    COMPARISON_STAGE = (
        "after_goldak"
    )


    # ========================================================
    # ANALYZE EACH GRAPH RESOLUTION
    # ========================================================

    mean_histories = {}


    for case in CASES:

        density = (
            case["node_density"]
        )


        case_runs = [

            run

            for run in all_runs

            if run["density"]
            == density
        ]


        # ----------------------------------------------------
        # Extract the same simulation stage from each run
        # ----------------------------------------------------

        stage_histories = []


        for run in case_runs:

            time_stage, T_stage = (
                extract_stage_history(
                    run,
                    COMPARISON_STAGE
                )
            )


            if len(time_stage) == 0:

                raise ValueError(
                    f"No '{COMPARISON_STAGE}' "
                    f"probe records found for "
                    f"{run['case']}, "
                    f"repeat {run['repeat']}."
                )


            stage_histories.append(
                (
                    time_stage,
                    T_stage
                )
            )


        # ----------------------------------------------------
        # Common time grid
        # ----------------------------------------------------

        common_time = (
            stage_histories[0][0]
        )


        temperature_matrix = np.vstack([

            np.interp(
                common_time,
                time_run,
                T_run
            )

            for (
                time_run,
                T_run
            ) in stage_histories

        ])


        # ----------------------------------------------------
        # Ignore positions where all realizations are NaN
        # ----------------------------------------------------

        mean_temperature = (
            np.nanmean(
                temperature_matrix,
                axis=0
            )
        )


        std_temperature = (
            np.nanstd(
                temperature_matrix,
                axis=0,
                ddof=1
            )
        )


        # ----------------------------------------------------
        # History scatter
        # ----------------------------------------------------

        finite_std = (
            np.isfinite(
                std_temperature
            )
        )


        if np.any(
            finite_std
        ):

            mean_history_std = (
                np.mean(
                    std_temperature[
                        finite_std
                    ]
                )
            )

            max_history_std = (
                np.max(
                    std_temperature[
                        finite_std
                    ]
                )
            )

        else:

            mean_history_std = (
                np.nan
            )

            max_history_std = (
                np.nan
            )


        # ----------------------------------------------------
        # Peak-temperature repeatability
        # ----------------------------------------------------

        peaks = np.array([

            run[
                "peak_temperature"
            ]

            for run in case_runs

        ])


        mean_peak = (
            np.mean(
                peaks
            )
        )


        std_peak = (
            np.std(
                peaks,
                ddof=1
            )
        )


        if mean_peak != 0:

            peak_cv = (
                std_peak
                / mean_peak
                * 100
            )

        else:

            peak_cv = (
                np.nan
            )


        # ----------------------------------------------------
        # Save mean history
        # ----------------------------------------------------

        d_string = (
            density_string(
                density
            )
        )


        mean_history_df = (
            pd.DataFrame({

                "Time (s)":
                    common_time,

                "Mean Temperature (C)":
                    mean_temperature,

                "Std Temperature (C)":
                    std_temperature,

                "Mean Minus 1 Std (C)":
                    (
                        mean_temperature
                        - std_temperature
                    ),

                "Mean Plus 1 Std (C)":
                    (
                        mean_temperature
                        + std_temperature
                    )
            })
        )


        mean_history_df.to_csv(

            DATA_DIR
            / (
                f"density_{d_string}"
                f"_mean_history.csv"
            ),

            index=False
        )


        mean_histories[
            case["name"]
        ] = {

            "time":
                common_time,

            "mean":
                mean_temperature,

            "std":
                std_temperature
        }


        # ----------------------------------------------------
        # Summary row
        # ----------------------------------------------------

        summary_results.append({

            "Case":
                case["name"],

            "Node Density (nodes/mm^3)":
                case["node_density"],

            "Epsilon (mm)":
                (
                    case["epsilon"]
                    * 1000
                ),

            "Gain":
                case["gain"],

            "Mean Total Nodes":
                np.mean([

                    run[
                        "total_nodes"
                    ]

                    for run
                    in case_runs
                ]),

            "Mean Peak Temperature (C)":
                mean_peak,

            "Peak Temperature Std (C)":
                std_peak,

            "Peak Temperature CV (%)":
                peak_cv,

            "Mean History Std (C)":
                mean_history_std,

            "Maximum History Std (C)":
                max_history_std,

            "Mean Graph Time (s)":
                np.mean([

                    run[
                        "graph_time"
                    ]

                    for run
                    in case_runs
                ]),

            "Mean Simulation Time (s)":
                np.mean([

                    run[
                        "simulation_computation_time"
                    ]

                    for run
                    in case_runs
                ]),

            "Mean Total Computation Time (s)":
                np.mean([

                    run[
                        "total_computation_time"
                    ]

                    for run
                    in case_runs
                ])
        })


    # ========================================================
    # SUMMARY TABLE
    # ========================================================

    summary_df = pd.DataFrame(
        summary_results
    )


    print(
        "\n"
        + "=" * 85
    )

    print(
        "ARTIFICIAL PROBE REPEATABILITY / "
        "GRAPH-RESOLUTION SUMMARY"
    )

    print(
        "=" * 85
    )


    print(
        summary_df.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df.to_csv(

        DATA_DIR
        / "probe_repeatability_summary.csv",

        index=False
    )


    summary_df.to_excel(

        DATA_DIR
        / "probe_repeatability_summary.xlsx",

        index=False
    )


    # ========================================================
    # PLOT 1
    # MEAN TEMPERATURE HISTORY
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )


    for case in CASES:

        case_name = (
            case["name"]
        )

        data = (
            mean_histories[
                case_name
            ]
        )


        plt.plot(
            data["time"],
            data["mean"],
            label=case_name
        )


    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Probe Temperature (°C)"
    )

    plt.title(
        "Artificial Melt-Pool Probe: "
        "Mean Temperature History"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR
        / "probe_mean_temperature_history.png",

        dpi=300
    )


    plt.show()


    # ========================================================
    # PLOT 2
    # REPEATABILITY SCATTER VS NODE DENSITY
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(

        summary_df[
            "Node Density (nodes/mm^3)"
        ],

        summary_df[
            "Mean History Std (C)"
        ],

        marker="o"
    )


    plt.xlabel(
        "Node Density (nodes/mm³)"
    )

    plt.ylabel(
        "Mean Temperature-History Std (°C)"
    )

    plt.title(
        "Random Point-Cloud Sensitivity "
        "vs Node Density"
    )

    plt.grid(
        True
    )

    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR
        / "probe_history_std_vs_density.png",

        dpi=300
    )


    plt.show()


    # ========================================================
    # PLOT 3
    # PEAK TEMPERATURE CV
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(

        summary_df[
            "Node Density (nodes/mm^3)"
        ],

        summary_df[
            "Peak Temperature CV (%)"
        ],

        marker="o"
    )


    plt.xlabel(
        "Node Density (nodes/mm³)"
    )

    plt.ylabel(
        "Peak Temperature CV (%)"
    )

    plt.title(
        "Peak-Temperature Repeatability "
        "vs Node Density"
    )

    plt.grid(
        True
    )

    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR
        / "probe_peak_cv_vs_density.png",

        dpi=300
    )


    plt.show()


    # ========================================================
    # PLOT 4
    # COMPUTATION COST
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )


    plt.plot(

        summary_df[
            "Node Density (nodes/mm^3)"
        ],

        summary_df[
            "Mean Total Computation Time (s)"
        ],

        marker="o"
    )


    plt.xlabel(
        "Node Density (nodes/mm³)"
    )

    plt.ylabel(
        "Mean Computation Time (s)"
    )

    plt.title(
        "Computational Cost "
        "vs Graph Resolution"
    )

    plt.grid(
        True
    )

    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR
        / "computation_time_vs_density.png",

        dpi=300
    )


    plt.show()


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":

    main()