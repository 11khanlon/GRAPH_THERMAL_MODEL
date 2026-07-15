"""
deposition.py

Main deposition simulation.

Implements

Section 4.3.1

Step 3
Step 4

Simulation order

for every block

    activate nodes

    apply Goldak heat source

    conduction

    convection

end block

after layer

    dwell conduction

    dwell convection

repeat until final layer
"""

import numpy as np

from conduction import ConductionSolver
from convection import ConvectionSolver
from heat_source import GoldakHeatSource


class DepositionSimulation:

    def __init__(
        self,
        geometry,
        nodes,
        graph,
        material,
        laser,
        convection,
        graph_settings,
    ):

        self.geometry = geometry
        self.nodes = nodes
        self.graph = graph

        self.material = material
        self.laser = laser
        self.convection = convection
        self.graph_settings = graph_settings

        self.temperature = np.full(
            len(nodes),
            material["ambient_temperature"],
            dtype=float,
        )

        self.history = []

        self.conduction = ConductionSolver(
            graph,
            gain=graph_settings["gain"],
        )

        self.cooling = ConvectionSolver(
            density=material["density"],
            block_length=convection["block_length"],
        )

        self.heat_source = GoldakHeatSource(
            laser_power=laser["power"],
            conductivity=material["thermal_conductivity"],
            diffusivity=material["thermal_diffusivity"],
            scan_speed=laser["scan_speed"],
            scaling_factor=laser["goldak_C"],
            meltpool_temperature=laser["meltpool_temperature"],
            liquidus_temperature=material["liquidus_temperature"],
        )

    ##########################################################

    def activate_block(self, block):
        """
        Activate nodes belonging to the
        newly deposited block.
        """

        for idx in block.node_indices:

            self.nodes[idx].active = True

    ##########################################################

    def surface_mask(self):

        return np.array(
            [node.surface for node in self.nodes],
            dtype=bool,
        )

    ##########################################################

    def active_temperature(self):

        T = self.temperature.copy()

        inactive = np.array(
            [not node.active for node in self.nodes]
        )

        T[inactive] = self.material["ambient_temperature"]

        return T

    ##########################################################

    def simulate_block(self, block):

        self.activate_block(block)

        ######################################################
        # Step 1
        # Laser heating
        ######################################################

        self.temperature = self.heat_source.apply(
            self.nodes,
            block.center,
            self.temperature,
        )

        ######################################################
        # Step 2
        # Conduction
        ######################################################

        alpha = self.material["thermal_diffusivity"]

        self.temperature = (
            self.conduction.block_conduction(
                self.temperature,
                alpha,
                self.laser["block_time"],
            )
        )

        ######################################################
        # Step 3
        # Convection
        ######################################################

        cp = self.material["specific_heat"]

        self.temperature = (
            self.cooling.block_cooling(
                self.temperature,
                self.surface_mask(),
                self.convection["forced"],
                cp,
                self.material["ambient_temperature"],
                self.laser["block_time"],
            )
        )

        self.history.append(
            self.temperature.copy()
        )

    ##########################################################

    def simulate_dwell(self, dwell_time):

        alpha = self.material["thermal_diffusivity"]

        cp = self.material["specific_heat"]

        steps = int(dwell_time)

        for _ in range(steps):

            self.temperature = (
                self.conduction.dwell_conduction(
                    self.temperature,
                    alpha,
                    1.0,
                )
            )

            self.temperature = (
                self.cooling.dwell_cooling(
                    self.temperature,
                    self.surface_mask(),
                    self.convection["forced"],
                    cp,
                    self.material["ambient_temperature"],
                    1.0,
                )
            )

            self.history.append(
                self.temperature.copy()
            )

    ##########################################################

    def run(self):

        print()

        print("----------------------------")
        print("Beginning deposition")
        print("----------------------------")

        current_layer = -1

        for block in self.geometry.blocks:

            if block.layer != current_layer:

                if current_layer != -1:

                    self.simulate_dwell(
                        self.laser["dwell_time"]
                    )

                current_layer = block.layer

                print(
                    f"Layer {current_layer+1}"
                )

            self.simulate_block(block)

        print()

        print("Simulation complete")

        self.history = np.array(
            self.history
        )

        return self.history

    ##########################################################

    def sensor_history(
        self,
        sensor_index,
    ):

        return self.history[:, sensor_index]

    ##########################################################

    def maximum_temperature(self):

        return np.max(
            self.temperature
        )

    ##########################################################

    def final_temperature(self):

        return self.temperature


##############################################################

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from graph import ThermalGraph

    from config import (
        MATERIAL,
        LASER,
        GRAPH,
        CONVECTION,
    )

    geometry = Geometry(
        "example_part.stl"
    )

    geometry.build_blocks()

    generator = NodeGenerator(
        geometry
    )

    generator.generate()

    graph = ThermalGraph(
        generator.nodes
    )

    graph.build()

    graph.degree_matrix()

    graph.laplacian()

    graph.eigensystem()

    simulation = DepositionSimulation(

        geometry,

        generator.nodes,

        graph,

        MATERIAL,

        LASER,

        CONVECTION,

        GRAPH,
    )

    history = simulation.run()

    print()

    print(
        "Maximum temperature:",
        simulation.maximum_temperature(),
    )

    print(
        "Timesteps:",
        history.shape[0],
    )