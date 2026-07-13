"""
main.py

Graph-Theory Thermal Simulation for Directed Energy Deposition

Implements the complete algorithm described in:

Section 4.3
Graph Theory Thermal Model for DED

Workflow

Step 1
    Build geometry

Step 2
    Generate graph
    Compute Laplacian eigensystem

Step 3
    Perform layer-by-layer deposition
        • Goldak heat source
        • Spectral conduction
        • Convection
        • Dwell cooling

Step 4
    Save thermal history

"""

from geometry import Geometry
from nodes import NodeGenerator
from graph import ThermalGraph
from deposition import DepositionSimulation
from visualization import (
    plot_sensor_temperature,
    plot_final_temperature
)


def main():

    print("\n========================================")
    print(" Graph-Theory DED Thermal Simulation")
    print("========================================\n")

    # --------------------------------------------------
    # Step 1
    # Build geometry
    # --------------------------------------------------

    geometry = Geometry("example_part.stl")

    geometry.build_blocks()

    print(
        f"Geometry created with "
        f"{len(geometry.blocks)} blocks."
    )

    # --------------------------------------------------
    # Step 2
    # Generate graph nodes
    # --------------------------------------------------

    generator = NodeGenerator(geometry)

    generator.generate()

    print(
        f"Generated {len(generator.nodes)} nodes."
    )

    # --------------------------------------------------
    # Step 3
    # Construct graph
    # --------------------------------------------------

    graph = ThermalGraph(generator.nodes)

    graph.build()

    graph.degree_matrix()

    graph.laplacian()

    graph.eigensystem()

    graph.summary()

    # --------------------------------------------------
    # Step 4
    # Run deposition simulation
    # --------------------------------------------------

    simulation = DepositionSimulation(

        geometry=geometry,
        nodes=generator.nodes,
        graph=graph

    )

    simulation.run()

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\nSimulation complete.")

    print(
        f"Recorded "
        f"{len(simulation.sensor_history)} "
        "temperature measurements."
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    plot_sensor_temperature(simulation)

    plot_final_temperature(simulation)


if __name__ == "__main__":

    main()