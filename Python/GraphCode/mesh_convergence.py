"""Node-density convergence study for the steel graph thermal model.

Place in Python/GraphCode alongside geometry.py, nodes.py, graph.py, deposition.py,
thermocouples.py, material.py, and config.py. Run: python mesh_convergence.py

Numerical convergence is measured against the finest simulated node density,
NOT against experimental TC data. Epsilon/gain are held fixed by default.
"""
from __future__ import annotations

import traceback
import subprocess
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import GRAPH
from simulation_runner import run_simulation, PROBE_NAMES

# ------------------------- USER SETTINGS -------------------------

# Density in nodes/mm^3. Start modestly; full graph eigensystems can be costly
NODE_DENSITIES = [.01, 0.1, 0.2, 0.35, 0.5, 0.6, 0.7]

# First study: fixed epsilon and gain from config.py.
# If True, epsilon scales with characteristic node spacing (different study).
SCALE_EPSILON_WITH_SPACING = False

# Only used when SCALE_EPSILON_WITH_SPACING=True.
# epsilon = EPSILON_SPACING_RATIO * density**(-1/3) [mm].
EPSILON_SPACING_RATIO = 3.0

RANDOM_SEED = 12345
PROBE_NAMES = ("TC_0", "TC_1", "TC_2", "TC_3")
MIN_COMMON_SAMPLES = 3

# Suggested screening tolerances, not established physical requirements.
RMSE_TOLERANCE_C = 2.0
PEAK_DIFFERENCE_TOLERANCE_C = 5.0

PROJECT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_DIR
    / "Validation"
    / "Results"
    / "MeshConvergence"
)


def epsilon_for_density(density):
    if SCALE_EPSILON_WITH_SPACING:
        return EPSILON_SPACING_RATIO * density ** (-1.0 / 3.0) * 1e-3
    return float(GRAPH["epsilon"])


def run_case(density):
    return run_simulation(
        node_density=density,
        epsilon=epsilon_for_density(density),
        gain=float(GRAPH["gain"]),
        seed=RANDOM_SEED)


def compare_to_reference(case, reference):
    """Interpolate each case on the shared reference time grid; never extrapolate."""
    a_t, a_y = case["time"], case["temperature"]
    b_t, b_y = reference["time"], reference["temperature"]
    common = b_t[(b_t >= a_t[0]) & (b_t <= a_t[-1])]
    if len(common) < MIN_COMMON_SAMPLES:
        raise ValueError("Insufficient overlapping simulation timestamps")
    row = {}
    errors = []
    for k, name in enumerate(PROBE_NAMES):
        a = np.interp(common, a_t, a_y[:, k])
        b = np.interp(common, b_t, b_y[:, k])
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < MIN_COMMON_SAMPLES:
            raise ValueError(f"Insufficient valid comparison samples for {name}")
        difference = a[valid] - b[valid]
        errors.extend(difference.tolist())
        row[f"{name}_RMSE_C"] = float(np.sqrt(np.mean(difference**2)))
        row[f"{name}_peak_difference_C"] = float(abs(np.max(a[valid]) - np.max(b[valid])))
    row["RMSE_vs_finest_C"] = float(np.sqrt(np.mean(np.square(errors))))
    row["max_peak_difference_C"] = max(
        row[f"{name}_peak_difference_C"] for name in PROBE_NAMES
    )
    row["passes_tolerance"] = (
        row["RMSE_vs_finest_C"] <= RMSE_TOLERANCE_C
        and row["max_peak_difference_C"] <= PEAK_DIFFERENCE_TOLERANCE_C
    )
    return row

def plot_error_vs_time(results):
    """
    Compare each coarser node density against the finest
    successfully simulated density over time.
    """

    reference = results[-1]
    ref_time = reference["time"]
    ref_temp = reference["temperature"]

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), sharex=True
    )

    for case in results[:-1]:

        case_time = case["time"]
        case_temp = case["temperature"]

        # Only compare overlapping timestamps
        valid_time = (
            (ref_time >= case_time[0])
            & (ref_time <= case_time[-1])
        )

        common_time = ref_time[valid_time]

        for k, ax in enumerate(axes.flat):

            # Interpolate coarser simulation to reference times
            interpolated = np.interp(
                common_time,
                case_time,
                case_temp[:, k]
            )

            reference_values = ref_temp[valid_time, k]

            # Signed temperature error
            error = interpolated - reference_values

            ax.plot(
                common_time,
                error,
                label=(
                    f"{case['density']:g} vs "
                    f"{reference['density']:g} nodes/mm³"
                )
            )

    for k, ax in enumerate(axes.flat):
        ax.axhline(0, color="black", linestyle="--",
                   linewidth=1)
        ax.set_title(PROBE_NAMES[k])
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Temperature difference (°C)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Mesh Convergence: Temperature Error vs Time",
        fontsize=14
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "temperature_error_vs_time.png",
        dpi=180
    )

    plt.close(fig)


def save_plots(results, summary):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for k, ax in enumerate(axes.flat):
        for case in results:
            ax.plot(case["time"], case["temperature"][:, k],
                    label=f'{case["density"]:g} nodes/mm³ ({case["n_nodes"]} nodes)')
        ax.set_title(PROBE_NAMES[k])
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUTPUT_DIR / "tc_histories_by_density.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["n_nodes"], summary["RMSE_vs_finest_C"], "o-")
    ax.axhline(RMSE_TOLERANCE_C, ls="--", label="Suggested RMSE tolerance")
    ax.set_xlabel("Actual total node count")
    ax.set_ylabel("RMSE versus finest simulation (°C)")
    ax.set_title("Node-density convergence (fixed graph settings unless configured)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence_rmse.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["n_nodes"], summary["total_seconds"], "o-")
    ax.set_xlabel("Actual total node count")
    ax.set_ylabel("Total runtime (s)")
    ax.set_title("Computational cost of refinement")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "runtime_vs_nodes.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(NODE_DENSITIES) < 2 or any(d <= 0 for d in NODE_DENSITIES):
        raise ValueError("Provide at least two positive NODE_DENSITIES")
    print("Steel material classes are configured in simulation_runner.py")
    print("IMPORTANT: Verify config.MATERIAL is also set to steel.")
    results, status_rows = [], []
    for density in sorted(set(map(float, NODE_DENSITIES))):
        print(f"\n--- Node density {density:g} nodes/mm³ ---", flush=True)
        try:
            case = run_case(density)
            results.append(case)
            pd.DataFrame({"time_s": case["time"], **{
                name: case["temperature"][:, k] for k, name in enumerate(PROBE_NAMES)
            }}).to_csv(OUTPUT_DIR / f"tc_history_density_{density:g}.csv", index=False)
            status_rows.append({"node_density": density, "status": "ok"})
        except Exception as exc:
            traceback.print_exc(limit=3)
            status_rows.append({"node_density": density, "status": f"failed: {exc}"})
        pd.DataFrame(status_rows).to_csv(OUTPUT_DIR / "case_status.csv", index=False)
    if len(results) < 2:
        raise RuntimeError("At least two successful simulations are required for convergence")
    reference = results[-1]
    summary_rows = []
    for case in results:
        row = {k: v for k, v in case.items() if k not in ("time", "temperature")}
        row.update(compare_to_reference(case, reference))
        row["reference_density"] = reference["density"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "mesh_convergence_summary.csv", index=False)
    save_plots(results, summary)
    plot_error_vs_time(results)
    # Finest reference always has zero self-error; it is not proof of convergence.
    candidates = summary.iloc[:-1]

    acceptable = candidates.loc[candidates["passes_tolerance"]]

    print("\nCONVERGENCE SUMMARY\n")
    print(summary.to_string(index=False))

    if acceptable.empty:
        print("\nNo tested coarser resolution met the convergence tolerances.")
        print("Graph calibration will NOT start.")
        print("Consider testing additional node densities.")
        return

    # Select lowest-density acceptable candidate
    choice = acceptable.sort_values("density").iloc[0]
    selected_density = float(choice["density"])

    print("\n" + "=" * 60)
    print("MESH CONVERGENCE COMPLETE")
    print(f"Selected node density: {selected_density:.6f} nodes/mm³")
    print(f"Actual nodes: {int(choice['n_nodes'])}")
    print(f"RMSE vs finest: {choice['RMSE_vs_finest_C']:.4f} °C")
    print("Starting graph calibration...")
    print("=" * 60)

    # Save selected density for calibrate_graph.py
    selected_file = OUTPUT_DIR / "selected_density.csv"

    pd.DataFrame([{
        "node_density": selected_density,
        "n_nodes": int(choice["n_nodes"]),
        "RMSE_vs_finest_C": float(choice["RMSE_vs_finest_C"]),
        "reference_density": float(choice["reference_density"]),
    }]).to_csv(selected_file, index=False)

    print(f"Selected density saved to: {selected_file}")

    # Automatically launch calibration
    calibration_script = Path(__file__).resolve().parent / "calibrate_graph.py"

    if not calibration_script.exists():
        raise FileNotFoundError(
            f"Calibration script not found: {calibration_script}"
        )

    subprocess.run(
        [sys.executable, str(calibration_script)],
        check=True,
        cwd=calibration_script.parent
    )


if __name__ == "__main__":
    log_dir = OUTPUT_DIR / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simulation_{timestamp}.txt"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    with open(log_file, "w", encoding="utf-8") as f:
        with redirect_stdout(Tee(sys.stdout, f)):
            with redirect_stderr(Tee(sys.stderr, f)):
                main()

    print(f"\nSimulation log saved to: {log_file}")
