"""Shared steel graph thermal simulation runner.

Place beside config.py, geometry.py, nodes.py, graph.py, deposition.py,
thermocouples.py and material.py in Python/GraphCode.

No aluminum fallback. Confirm class names and config.MATERIAL steel values.
Each call constructs a fresh model to avoid stale deposition/node state.
"""
from __future__ import annotations

import importlib
import time
import numpy as np
from scipy.sparse.csgraph import connected_components

from config import MATERIAL, GRAPH, CONVECTION, BLOCK, DWELL, BUILD, SENSORS, WAAM
from geometry import Geometry
from nodes import NodeGenerator
from graph import ThermalGraph
from deposition import DepositionSimulation
from thermocouples import ThermalProbe, check_probe_location
from material import A36Material, WireMaterial, SubstrateMaterial
from config import load_experimental_dwell, DWELL

# EDIT these if the classes in your material.py use different names.
SUBSTRATE_MATERIAL_CLASS = "A36Material"
WIRE_MATERIAL_CLASS = "A36Material"
PROBE_NAMES = ("TC_0", "TC_1", "TC_2", "TC_3")
DEFAULT_SEED = 12345


def extract_probe_history(simulation):
    tracker = simulation.tracker
    histories = getattr(tracker, "probe_history", None)
    if histories is None:
        histories = getattr(tracker, "probe_temperature_histories", None)
    times = getattr(tracker, "probe_history_time", None)
    if times is None:
        times = getattr(tracker, "probe_time", None)
    if histories is None or times is None:
        raise AttributeError("Missing tracker probe history or times")
    times = np.asarray(times, dtype=float)
    temperatures = np.column_stack([np.asarray(histories[name], dtype=float) for name in PROBE_NAMES])
    if temperatures.shape != (len(times), len(PROBE_NAMES)):
        raise ValueError("Probe histories do not match timestamp array")
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Probe times must be finite and nondecreasing")
    # Preserve the last recorded state for duplicate timestamps.
    unique_times, reversed_indices = np.unique(times[::-1], return_index=True)
    indices = len(times) - 1 - reversed_indices
    return unique_times, temperatures[indices]


def run_simulation(node_density=None, epsilon=None, gain=None, seed=DEFAULT_SEED):
    """Return a dict with time, temperature, and graph diagnostics.

    node_density: nodes/mm^3; epsilon: meters; gain: same units as config.GRAPH.
    No optimization or experimental data access occurs here.
    """
    load_experimental_dwell()
    original_graph = dict(GRAPH)
    original_rng = np.random.get_state()
    started = time.perf_counter()
    try:
        settings = dict(GRAPH)
        if node_density is not None:
            settings["node_density"] = float(node_density)
        if epsilon is not None:
            settings["epsilon"] = float(epsilon)
        if gain is not None:
            settings["gain"] = float(gain)
        if settings["node_density"] <= 0 or settings["epsilon"] <= 0 or settings["gain"] <= 0:
            raise ValueError("node_density, epsilon and gain must be positive")
        # Existing NodeGenerator and ThermalGraph read the global GRAPH dictionary.
        GRAPH.update(settings)
        if seed is not None:
            np.random.seed(seed)
        geometry = Geometry(BUILD["stl_file"], substrate_material= SubstrateMaterial(),
                            wire_material= WireMaterial())
        geometry.build_blocks()
        generator = NodeGenerator(geometry)
        generator.generate()
        nodes = generator.nodes
        generator.validate_nodes()
        generator.print_surface_statistics() 
        if not nodes:
            raise ValueError("Node generator returned zero nodes")
        probes = [ThermalProbe(name=name,
                  position=np.array([sensor["x"], sensor["y"],
                                     geometry.substrate_top + sensor["z"]]),
                  region=sensor["region"]) for name, sensor in SENSORS.items()]
        if set(PROBE_NAMES) != {p.name for p in probes}:
            raise ValueError(f"Expected sensor names {PROBE_NAMES}; got {[p.name for p in probes]}")
        graph = ThermalGraph(nodes)
        graph.build()
        components, labels = connected_components(graph.A, directed=False)
        degrees = np.diff(graph.A.indptr)
        graph.degree_matrix()
        graph.laplacian()
        graph.eigensystem()
        graph_seconds = time.perf_counter() - started
        simulation = DepositionSimulation(
            geometry=geometry, node_generator=generator, graph=graph,
            material=MATERIAL, convection=CONVECTION, graph_settings=settings,
            dwell_info=DWELL, dwell_time=DWELL["dwell"],
            block_info=BLOCK, build_info=BUILD, probes=probes, waam=WAAM)
        simulation.run()
        times, temperatures = extract_probe_history(simulation)
        return {
            "density": float(settings["node_density"]),
            "epsilon_m": float(settings["epsilon"]),
            "gain": float(settings["gain"]),
            "n_nodes": len(nodes),
            "n_components_full_graph": int(components),
            "largest_component_fraction": float(np.bincount(labels).max() / len(nodes)),
            "isolated_nodes": int(np.count_nonzero(degrees == 0)),
            "mean_neighbors": float(degrees.mean()),
            "graph_seconds": graph_seconds,
            "total_seconds": time.perf_counter() - started,
            "time": times,
            "temperature": temperatures,
        }
    finally:
        GRAPH.clear()
        GRAPH.update(original_graph)
        np.random.set_state(original_rng)


if __name__ == "__main__":
    print("\nRunning single steel simulation...")

    results = run_simulation()

    print("\nSimulation completed successfully!")
    print(f"Nodes: {results['n_nodes']}")
    print(f"Connected components: {results['n_components_full_graph']}")
    print(f"Isolated nodes: {results['isolated_nodes']}")
    print(f"Total runtime: {results['total_seconds']:.2f} seconds")

    for i, name in enumerate(PROBE_NAMES):
        print(
            f"{name}: Peak temperature = "
            f"{np.nanmax(results['temperature'][:, i]):.2f} °C"
        )