"""
Step 5

Implements Goldak-based laser heating for the graph DED model

Based on Section 4.3.3 of

This module computes the initial temperature T0
for nodes inside the newly deposited block and
adjacent subsurface blocks.
"""

import numpy as np

from Python.config import LASER
from Python.config import MATERIAL


class GoldakHeatSource:

    """
    Goldak double-ellipsoid inspired heat source

    The paper uses Goldak's model to estimate
    initial node temperatures before graph diffusion
    """

    def __init__(self):

        self.power = LASER["power"]

        self.scan_speed = LASER["scan_speed"]

        self.beam_radius = LASER["beam_diameter"] / 2

        self.k = MATERIAL["thermal_conductivity"]

        self.ambient = MATERIAL["ambient_temperature"]

        self.meltpool_temperature = LASER["meltpool_temperature"]

        self.liquidus = MATERIAL["liquidus_temperature"]

        # Characteristic decay lengths (meters)
        # These are tunable and roughly correspond
        # to melt pool dimensions.

        self.ax = 0.75e-3
        self.by = 0.75e-3
        self.cz = 1.25e-3

    # --------------------------------------------------------

    def temperature(self,
                    x,
                    y,
                    z):
        """
        Initial temperature field.

        Returns T0(x,y,z)

        centered on laser position (0,0,0)
        """

        r = (
            (x/self.ax)**2
            +
            (y/self.by)**2
            +
            (z/self.cz)**2
        )

        return (
            self.ambient
            +
            (self.meltpool_temperature-self.ambient)
            *
            np.exp(-3*r)
        )

    # --------------------------------------------------------

    def heat_block(self,
                   block,
                   laser_position):
        """
        Assign temperatures to every node
        inside one deposited block.
        """

        lx, ly, lz = laser_position

        for node in block.nodes:

            x = node.position[0] - lx
            y = node.position[1] - ly
            z = node.position[2] - lz

            node.temperature = self.temperature(
                x,
                y,
                z
            )

            node.active = True

    # --------------------------------------------------------

    def heat_subsurface(
            self,
            current_layer,
            geometry,
            laser_position,
            depth_layers=7):
        """
        Reheat underlying material.

        Paper:
        laser penetrates roughly
        seven previous layers.
        """

        lx, ly, lz = laser_position

        for block in geometry.blocks:

            if not block.active:
                continue

            dz = current_layer - block.layer

            if dz <= 0:
                continue

            if dz > depth_layers:
                continue

            attenuation = np.exp(
                -dz/depth_layers
            )

            for node in block.nodes:

                x = node.position[0] - lx
                y = node.position[1] - ly
                z = node.position[2] - lz

                T = self.temperature(
                    x,
                    y,
                    z
                )

                node.temperature = max(
                    node.temperature,
                    self.ambient +
                    attenuation *
                    (T-self.ambient)
                )

    # --------------------------------------------------------

    def apply(
            self,
            block,
            geometry):
        """
        Complete heating operation.

        1) Heat current block.

        2) Reheat previous layers.
        """

        laser_position = block.center

        self.heat_block(
            block,
            laser_position
        )

        self.heat_subsurface(
            block.layer,
            geometry,
            laser_position
        )


# ------------------------------------------------------------

if __name__ == "__main__":

    from Python.geometry import Geometry
    from Python.nodes import NodeGenerator

    geom = Geometry("example_part.stl")

    geom.build_blocks()

    gen = NodeGenerator(geom)

    gen.generate()

    heater = GoldakHeatSource()

    block = geom.blocks[0]

    heater.apply(
        block,
        geom
    )

    print(block.nodes[0].temperature)