"""
heat_source.py

Laser heat source model.

Implements Section 4.3.3

Goldak's double ellipsoid model
(Paper Equation 19)

Produces the initial nodal temperature field
before conduction begins.

This module ONLY applies laser heating.

No conduction.
No convection.
No graph operations.
"""

import numpy as np


class GoldakHeatSource:

    def __init__(
        self,
        laser_power,
        conductivity,
        diffusivity,
        scan_speed,
        scaling_factor,
        meltpool_temperature,
        liquidus_temperature,
    ):

        self.P = laser_power
        self.k = conductivity
        self.alpha = diffusivity
        self.V = scan_speed
        self.C = scaling_factor

        self.Tmelt = meltpool_temperature
        self.Tliquidus = liquidus_temperature

    # --------------------------------------------------------

    def goldak_temperature(
        self,
        x,
        y,
        z,
    ):
        """
        Equation (19)

        Returns laser-induced temperature
        at local coordinates (x,y,z).
        """

        r = np.sqrt(
            x**2 +
            y**2 +
            z**2
        )

        # avoid singularity

        r = np.maximum(
            r,
            1e-8
        )

        temperature = (
            self.C
            * self.P
            /
            (
                2
                * np.pi
                * self.k
                * r
            )
            *
            np.exp(
                -self.V
                /
                (
                    2
                    * self.alpha
                )
                *
                (
                    x + r
                )
            )
        )

        return temperature

    # --------------------------------------------------------

    def heat_block(
        self,
        nodes,
        block_center,
        temperature,
    ):
        """
        Heat one active block.

        Parameters
        ----------
        nodes

        block_center

        temperature

        Returns
        -------
        Updated temperature vector.
        """

        T = temperature.copy()

        xc, yc, zc = block_center

        for i, node in enumerate(nodes):

            if not node.active:
                continue

            x = node.position[0] - xc
            y = node.position[1] - yc
            z = node.position[2] - zc

            delta = self.goldak_temperature(
                x,
                y,
                z,
            )

            T[i] = max(
                T[i],
                min(
                    delta,
                    self.Tmelt
                )
            )

        return T

    # --------------------------------------------------------

    def heat_subsurface(
        self,
        nodes,
        block_center,
        temperature,
        cutoff=0.20,
    ):
        """
        Heat underlying layers.

        Paper considers reheating down
        to approximately

        20%

        of liquidus temperature.
        """

        T = temperature.copy()

        threshold = (
            cutoff
            * self.Tliquidus
        )

        xc, yc, zc = block_center

        for i, node in enumerate(nodes):

            if not node.active:
                continue

            x = node.position[0] - xc
            y = node.position[1] - yc
            z = node.position[2] - zc

            delta = self.goldak_temperature(
                x,
                y,
                z,
            )

            if delta >= threshold:

                T[i] = max(
                    T[i],
                    delta
                )

        return T

    # --------------------------------------------------------

    def apply(
        self,
        nodes,
        block_center,
        temperature,
    ):
        """
        Complete laser heating.

        Surface
        +
        subsurface heating.
        """

        T = self.heat_block(
            nodes,
            block_center,
            temperature,
        )

        T = self.heat_subsurface(
            nodes,
            block_center,
            T,
        )

        return T


# ------------------------------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from config import LASER, MATERIAL

    geom = Geometry("example_part.stl")

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    nodes = generator.nodes

    heat = GoldakHeatSource(

        laser_power=LASER["power"],

        conductivity=MATERIAL["thermal_conductivity"],

        diffusivity=MATERIAL["thermal_diffusivity"],

        scan_speed=LASER["scan_speed"],

        scaling_factor=LASER["goldak_C"],

        meltpool_temperature=LASER["meltpool_temperature"],

        liquidus_temperature=MATERIAL["liquidus_temperature"],
    )

    temperature = np.full(
        len(nodes),
        MATERIAL["ambient_temperature"],
    )

    block = geom.blocks[0]

    center = block.center

    heated = heat.apply(
        nodes,
        center,
        temperature,
    )

    print("Maximum temperature:", heated.max())