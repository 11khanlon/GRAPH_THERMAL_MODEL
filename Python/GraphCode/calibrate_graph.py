"""Calibrate graph neighborhood radius and spectral conduction gain against steel Pass 1 TC data.

Run from your project environment: python calibrate_graph.py
This script uses simulation_runner.py and mesh_convergence.py output.
"""
from pathlib import Path
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import GRAPH
from simulation_runner import run_simulation, PROBE_NAMES

TC_COLUMNS = ("ch0", "ch1", "ch2", "ch3")
TC_NAMES = PROBE_NAMES

EPSILON_VALUES = np.array([1e-3, 3.0e-3, 4.0e-3])
GAIN_VALUES = np.array([ 3e6, 1e7, 1e8]) 
'''EPSILON_VALUES = np.array([1, 3.0, 4.0, 4.75, 5.5, 6.5, 12, 30, 35]) * 1e-3  # meters
GAIN_VALUES = np.array([1e5, 3e5, 1e6, 3e6, 1e7, 1e8])       # m^-2'''

# At least this fraction of the baseline's valid experimental comparison samples must remain comparable in each candidate. No time-delay fitting is performed
MIN_COVERAGE = 0.95
# Set to an integer only if your node generator uses NumPy's global random state.
NODE_SEED = 12345

from pathlib import Path
import pandas as pd

def selected_density():
    summary_file = (
        Path(__file__).resolve().parents[2]
        / "Validation"
        / "Results"
        / "MeshConvergence"
        / "mesh_convergence_summary.csv"
    )

    df = pd.read_csv(summary_file)

    # Exclude the finest mesh (zero self-error)
    candidates = df.sort_values("density").iloc[:-1]

    # Find candidates that passed convergence
    acceptable = candidates.loc[
        candidates["passes_tolerance"]
        .astype(str)
        .str.lower()
        .eq("true")
    ]

    if acceptable.empty:
        raise ValueError(
            "No node density passed the convergence tolerances."
        )

    # Select lowest acceptable density
    density = float(acceptable.iloc[0]["density"])

    print(f"Selected node density: {density:.6f} nodes/mm³")

    return density



PROJECT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_DIR
    / "Validation"
    / "Results"
    / "Calibration"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_data_file():
    """Find the supplied Pass 1 CSV under a project parent directory."""
    checked = []
    for root in (PROJECT_DIR, *list(PROJECT_DIR.parents)[:4]):
        for pass_folder in ("Pass_1", "Pass1"):
            candidate = (root / "A36Steel_Info" / "ThermalData" /
                         "Processed" / pass_folder / "Thermocouples_Pass1.csv")
            checked.append(candidate)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("Thermocouples_Pass1.csv not found. Checked:\n" +
                            "\n".join(str(p) for p in checked))


def load_experimental_data():
    path = find_data_file()
    df = pd.read_csv(path)
    required = ["RelativeTime_s", "IsDepositing", *TC_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df["RelativeTime_s"] = pd.to_numeric(df["RelativeTime_s"], errors="coerce")
    for col in TC_COLUMNS:
        # Boolean FALSE and blank entries are missing, NOT 0 degrees Celsius.
        df[col] = pd.to_numeric(
            df[col].replace({True: np.nan, False: np.nan,
                             "TRUE": np.nan, "FALSE": np.nan,
                             "true": np.nan, "false": np.nan}),
            errors="coerce",
        )
    df["IsDepositing"] = (df["IsDepositing"].astype(str).str.strip()
                          .str.lower().isin(("true", "1")))
    df = df.loc[df["RelativeTime_s"].notna() &
                (df["RelativeTime_s"] >= 0)].copy()
    df = (df.groupby("RelativeTime_s", as_index=False)
          .agg({**{col: "mean" for col in TC_COLUMNS}, "IsDepositing": "max"})
          .sort_values("RelativeTime_s").reset_index(drop=True))
    print(f"Experimental file: {path}")
    for col in TC_COLUMNS:
        print(f"  {col}: {df[col].notna().sum()} valid readings")
    if not any(df[col].notna().any() for col in TC_COLUMNS):
        raise ValueError("No usable experimental thermocouple measurements.")
    return df


def compare_temperatures(experimental, sim_time, sim_temperature,
                         reference_counts=None):
    """RMSE with fixed time alignment and a coverage check per channel."""
    exp_time = experimental["RelativeTime_s"].to_numpy(dtype=float)
    sim_time = np.asarray(sim_time, dtype=float)
    sim_temperature = np.asarray(sim_temperature, dtype=float)
    if sim_temperature.shape != (len(sim_time), 4):
        raise ValueError("Simulation temperatures must have shape (N, 4)")
    if len(sim_time) < 2 or np.any(np.diff(sim_time) <= 0):
        raise ValueError("Simulation times must be strictly increasing")

    all_errors = []
    counts = []
    for k, channel in enumerate(TC_COLUMNS):
        measured = experimental[channel].to_numpy(dtype=float)
        # Keep a fixed experimental window: all measured samples between
        # simulation start and the experiment's last timestamp.
        reference_mask = np.isfinite(measured) & (exp_time >= 0)
        reference_count = int(reference_mask.sum())
        if reference_count == 0:
            counts.append(0)
            continue
        finite = np.isfinite(sim_temperature[:, k])
        if finite.sum() < 2:
            return np.inf, counts + [0], reference_count
        valid = (reference_mask & (exp_time >= sim_time[0]) &
                 (exp_time <= sim_time[-1]))
        # Do not silently bridge invalid simulated samples.
        query = exp_time[valid]
        if query.size == 0:
            return np.inf, counts + [0], reference_count
        predicted = np.interp(query, sim_time, sim_temperature[:, k])
        ok = np.isfinite(predicted)
        count = int(ok.sum())
        counts.append(count)
        needed = (reference_counts[k] if reference_counts is not None
                  else reference_count)
        if needed > 0 and count < MIN_COVERAGE * needed:
            return np.inf, counts, reference_count
        if count:
            all_errors.extend((predicted[ok] - measured[valid][ok]).tolist())
    if not all_errors:
        return np.inf, counts, 0
    return float(np.sqrt(np.mean(np.square(all_errors)))), counts, len(all_errors)


# Mesh convergence must be run first. This is a numerical-resolution choice,
# not an optimization over experimental error.
MESH_SUMMARY = PROJECT_DIR / "Validation" / "Results" / "MeshConvergence" / "mesh_convergence_summary.csv"
NODE_DENSITY = None  # Set an explicit density to override automatic selection.


def per_tc_rmse(experimental, sim_time, sim_temperature):
    """Diagnostics only; original pooled-RMSE objective is unchanged."""
    exp_time = experimental["RelativeTime_s"].to_numpy(dtype=float)
    metrics = {}
    for k, channel in enumerate(TC_COLUMNS):
        measured = experimental[channel].to_numpy(dtype=float)
        valid = (np.isfinite(measured) & (exp_time >= sim_time[0])
                 & (exp_time <= sim_time[-1]))
        if not np.any(valid):
            metrics[f"{channel}_RMSE_C"] = np.nan
            continue
        predicted = np.interp(exp_time[valid], sim_time, sim_temperature[:, k])
        valid_pairs = np.isfinite(predicted)
        errors = predicted[valid_pairs] - measured[valid][valid_pairs]
        metrics[f"{channel}_RMSE_C"] = (float(np.sqrt(np.mean(errors**2)))
                                         if len(errors) else np.nan)
    return metrics


def run_candidate(epsilon, gain, density):
    """Use the shared runner, rather than duplicating graph/deposition setup."""
    result = run_simulation(node_density=density, epsilon=epsilon, gain=gain,
                            seed=NODE_SEED)
    # Preserve the original calibration behavior: no graph-connectivity filter.
    return result["time"], result["temperature"]


def save_best_fit_plot(experimental, sim_time, sim_temperature, epsilon, gain, rmse):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for k, ax in enumerate(axes.flat):
        ax.plot(experimental["RelativeTime_s"], experimental[TC_COLUMNS[k]],
                label="Experimental", color="black", lw=1.3)
        ax.plot(sim_time, sim_temperature[:, k],
                label="Simulated", color="tab:red", lw=1.3)
        ax.set_title(f"{TC_NAMES[k]} / {TC_COLUMNS[k]}")
        ax.set_xlabel("Time since deposition start (s)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Best graph fit: ε={epsilon * 1e3:.3f} mm, "
                 f"g={gain:.3g} m⁻², RMSE={rmse:.3f} °C")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_fit_thermocouples.png", dpi=200)
    plt.close(fig)


def save_rmse_heatmap(results):
    table = results.pivot(index="epsilon_mm", columns="gain", values="RMSE_C")
    fig, ax = plt.subplots(figsize=(9, 5))
    values = table.to_numpy(dtype=float)
    image = ax.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto")
    ax.set_xticks(range(len(table.columns)), [f"{v:.1e}" for v in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{v:g}" for v in table.index])
    ax.set_xlabel("Gain g (m⁻²)")
    ax.set_ylabel("Neighborhood radius ε (mm)")
    ax.set_title("Thermocouple RMSE (°C)")
    fig.colorbar(image, ax=ax, label="RMSE (°C)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rmse_heatmap.png", dpi=200)
    plt.close(fig)


def calibrate_graph(experimental, density):
    rows = []
    best = None
    # Require each candidate to cover nearly all recorded samples for each TC.
    reference_counts = [int(experimental[c].notna().sum()) for c in TC_COLUMNS]
    for epsilon in EPSILON_VALUES:
        for gain in GAIN_VALUES:
            print(f"\nTesting epsilon={epsilon * 1e3:.3f} mm; gain={gain:.3e}", flush=True)
            try:
                times, temperatures = run_candidate(epsilon, gain, density)
                rmse, counts, n_compared = compare_temperatures(
                    experimental, times, temperatures, reference_counts
                )
                status = "ok" if np.isfinite(rmse) else "insufficient coverage"
                if np.isfinite(rmse) and (best is None or rmse < best[0]):
                    best = (rmse, epsilon, gain, times.copy(), temperatures.copy())
            except Exception as exc:
                print(f"Candidate failed: {exc}")
                traceback.print_exc(limit=2)
                rmse, counts, n_compared = np.inf, [], 0
                status = f"error: {type(exc).__name__}: {exc}"
            tc_metrics = (per_tc_rmse(experimental, times, temperatures)
                          if status in ("ok", "insufficient coverage") else {})
            rows.append({
                "node_density": density,
                "epsilon_m": epsilon,
                "epsilon_mm": epsilon * 1e3,
                "gain": gain,
                "RMSE_C": rmse,
                "samples_compared": n_compared,
                "status": status,
                **tc_metrics,
                **{f"n_{name}": counts[k] if k < len(counts) else 0
                   for k, name in enumerate(TC_COLUMNS)},
            })
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "graph_calibration_results.csv", index=False)
            print(f"RMSE={rmse:.4f} °C | {status}", flush=True)
    results = pd.DataFrame(rows).sort_values("RMSE_C", kind="stable")
    if best is not None:
        rmse, epsilon, gain, times, temperatures = best
        save_best_fit_plot(experimental, times, temperatures, epsilon, gain, rmse)
        np.savez_compressed(OUTPUT_DIR / "best_fit_probe_history.npz",
                            time=times, temperature=temperatures,
                            epsilon=epsilon, gain=gain, rmse=rmse)
    if results["RMSE_C"].notna().any() and np.isfinite(results["RMSE_C"]).any():
        save_rmse_heatmap(results)
    return results


if __name__ == "__main__":
    experimental = load_experimental_data()
    density = selected_density()
    print(f"Using node density: {density:g} nodes/mm³")
    results = calibrate_graph(experimental, density)
    print("\nBEST CANDIDATES")
    print(results.head(10).to_string(index=False))
    print(f"\nResults directory: {OUTPUT_DIR}")
