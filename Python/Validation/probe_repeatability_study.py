"""
Artificial probe repeatability and node-density convergence study.

The probe is placed at a fixed physical location near the melt pool.
This is NOT the experimental Case-A thermocouple.

Three randomized node realizations are run for each graph density.

Goals:
    1. Quantify sensitivity to random node placement.
    2. Determine whether higher node density reduces temperature scatter.
    3. Compare mean temperature histories between graph resolutions.
"""

import sys
from pathlib import Path


# PROJECT PATHS

VALIDATION_DIR = Path(__file__).resolve().parent

PYTHON_DIR = VALIDATION_DIR.parent

# Allow modules in Python/ to be imported
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_DIR)
    )

# RESULTS PATHS

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

import time
from pathlib import Path

import numpy as np
import pandas as pd

from geometry import Geometry
from nodes import NodeGenerator
from graph import ThermalGraph
from deposition import DepositionSimulation

from config import (
    MATERIAL,
    LASER,
    GRAPH,
    CONVECTION,
    BLOCK,
    DWELL,
    BUILD
)


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

all_runs = []
summary_results = []

for case in CASES:

    for repeat in range(1, N_REPEATS + 1):

        print(
            f"\nRunning {case['name']} "
            f"- Repeat {repeat}/{N_REPEATS}"
        )

        GRAPH["node_density"] = (
            case["node_density"]
        )

        GRAPH["epsilon"] = (
            case["epsilon"]
        )

        GRAPH["gain"] = (
            case["gain"]
        )

        total_start = time.perf_counter()

        geometry = Geometry(
            BUILD["stl_file"]
        )

        geometry.build_blocks()

        # New random point cloud every time
        generator = NodeGenerator(
            geometry
        )

        generator.generate()

        graph = ThermalGraph(
            generator.nodes
        )

        probe_node, distance = (
            graph.find_sensor_node(
                geometry.sensor_position
            )
        )

        # Put probe at exact same physical location
        probe_node.position = (
            geometry.sensor_position.copy()
        )

        graph_start = time.perf_counter()
        graph.build()
        graph.degree_matrix()
        graph.laplacian()
        graph.eigensystem()

        graph_time = (
            time.perf_counter()
            - graph_start
        )


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
            block_info=BLOCK
        )

        simulation_start = time.perf_counter()

        simulation.run()

        simulation_time_wall = (
            time.perf_counter()
            - simulation_start
        )

        total_computation_time = (
            time.perf_counter()
            - total_start
        )

        probe_temperature = (
            simulation.sensor_history(
                probe_node.id
            )
        )

        density_string = (
            str(case["node_density"])
            .replace(".", "p")
        )

        run_df = pd.DataFrame({
            "Time (s)": simulation.history_time,
            "Probe Temperature (C)": probe_temperature
        })

        run_df.to_csv(
            DATA_DIR
            / (
                f"density_{density_string}"
                f"_repeat_{repeat}.csv"
            ),
            index=False
        )

        all_runs.append({
            "case": case["name"],
            "density": case["node_density"],
            "epsilon": case["epsilon"],
            "gain": case["gain"],
            "repeat": repeat,
            "time": simulation.history_time.copy(),
            "temperature": probe_temperature.copy(),
            "peak_temperature":
                np.max(probe_temperature),
            "graph_time":
                graph_time,

            "simulation_computation_time":
                simulation_time_wall,

            "total_computation_time":
                total_computation_time,

            "total_nodes":
                len(generator.nodes)
        })

    

for case in CASES:

    density = case["node_density"]

    case_runs = [
        run
        for run in all_runs
        if run["density"] == density
    ]

    common_time = (
        case_runs[0]["time"]
    )

    temperature_matrix = np.vstack([
        np.interp(
            common_time,
            run["time"],
            run["temperature"]
        )
        for run in case_runs
    ])

    mean_temperature = np.mean(
        temperature_matrix,
        axis=0
    )

    std_temperature = np.std(
        temperature_matrix,
        axis=0,
        ddof=1
    )

    mean_history_std = np.mean(
    std_temperature
    )

    max_history_std = np.max(
        std_temperature
    )

    peaks = np.array([
        run["peak_temperature"]
        for run in case_runs
    ])

    mean_peak = np.mean(peaks)

    std_peak = np.std(
        peaks,
        ddof=1
    )

    if mean_peak != 0:
        peak_cv = (
            std_peak
            / mean_peak
            * 100
        )
    else:
        peak_cv = np.nan

    density_string = (
            str(density)
            .replace(".", "p")
        )

    mean_history_df = pd.DataFrame({
        "Time (s)": common_time,
        "Mean Temperature (C)": mean_temperature,
        "Std Temperature (C)": std_temperature,
        "Mean Minus 1 Std (C)":
            mean_temperature - std_temperature,
        "Mean Plus 1 Std (C)":
            mean_temperature + std_temperature
    })

    mean_history_df.to_csv(
        DATA_DIR
        / f"density_{density_string}_mean_history.csv",
        index=False
    )


    summary_results.append({

            "Case":
                case["name"],

            "Node Density (nodes/mm^3)":
                case["node_density"],

            "Epsilon (mm)":
                case["epsilon"] * 1000,

            "Gain":
                case["gain"],

            "Mean Total Nodes":
                np.mean([
                    run["total_nodes"]
                    for run in case_runs
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
                    run["graph_time"]
                    for run in case_runs
                ]),

            "Mean Simulation Time (s)":
                np.mean([
                    run["simulation_computation_time"]
                    for run in case_runs
                ]),

            "Mean Total Computation Time (s)":
                np.mean([
                    run["total_computation_time"]
                    for run in case_runs
                ]),
        })


summary_df = pd.DataFrame(
    summary_results
)

print(
    "\n"
    + "=" * 75
)

print(
    "PROBE REPEATABILITY / MESH CONVERGENCE SUMMARY"
)

print(
    "=" * 75
)

print(
    summary_df.to_string(
        index=False
    )
)


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
 



  