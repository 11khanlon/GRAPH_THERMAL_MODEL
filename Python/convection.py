"""
Heat loss through convection (and radiation approximation)

Implements Section 4.3.1.5 and Section 4.3.1.6

Paper Equations:
(16): Tb = Tc exp(-β tb)
(18): TLf = TLc exp(-β t)

where

β = h / rho L Cp

This module only computes boundary cooling
No graph operations are performed here
"""

import numpy as np


class ConvectionSolver:

    """
    Newton cooling model

    Convection and radiation are lumped into an
    single effective heat transfer coefficient.
    """

    def __init__(
        self,
        density,
        block_length
    ):
        """
        Parameters:
        density : float , Material density (kg/m^3)
        block_length : float, Characteristic block length (m)
        """

        self.rho = density
        self.L = block_length

    # -------------------------------------------------------

    def beta(
        self,
        h,
        cp):

        """
        Compute inverse time constant

            β = h / (rho L Cp)

        Parameters:
        h : float, Heat transfer coefficient
        cp : float,  Specific heat

        Returns:
        float
        """

        return h / (
            self.rho
            * self.L
            * cp
        )

    # -------------------------------------------------------

    def cool(
        self,
        temperature,
        surface_mask,
        h,
        cp,
        ambient_temperature,
        dt
    ):
        """
        Apply Newton cooling to surface nodes
        Implements Equation (16)

        Parameters:
        temperature : ndarray
        surface_mask : ndarray(bool), True for boundary nodes

        h : float
        cp : float
        ambient_temperature : float
        dt : float

        Returns:
        ndarray
        """

        beta = self.beta(
            h,
            cp,
        )

        T = temperature.copy()

        factor = np.exp(
            -beta * dt
        )

        boundary = np.where(surface_mask)[0]

        T[boundary] = (
            ambient_temperature
            + (
                T[boundary]
                - ambient_temperature
            )
            * factor
        )

        return T

    # -------------------------------------------------------

    def block_cooling(
        self,
        temperature,
        surface_mask,
        h,
        cp,
        ambient_temperature,
        block_time
    ):
        """
        Equation (16)

        Cooling immediately after one block.
        """

        return self.cool(
            temperature,
            surface_mask,
            h,
            cp,
            ambient_temperature,
            block_time
        )

    # -------------------------------------------------------

    def dwell_cooling(
        self,
        temperature,
        surface_mask,
        h,
        cp,
        ambient_temperature,
        dt=1.0
    ):
        """
        Equation (18)

        Cooling during dwell time.

        One call corresponds to
        one second.
        """

        return self.cool(
            temperature,
            surface_mask,
            h,
            cp,
            ambient_temperature,
            dt,
        )

    # -------------------------------------------------------

    def multiple_steps(
        self,
        temperature,
        surface_mask,
        h,
        cp,
        ambient_temperature,
        total_time,
        dt=1.0
    ):
        """
        Repeated convection cooling.

        Used for arbitrary dwell periods.
        """

        T = temperature.copy()

        steps = int(
            np.ceil(total_time / dt)
        )

        for _ in range(steps):

            T = self.cool(
                T,
                surface_mask,
                h,
                cp,
                ambient_temperature,
                dt
            )

        return T


# ------------------------------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from config import MATERIAL, CONVECTION, BLOCK

    geom = Geometry("example_part.stl")

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    nodes = generator.nodes

    temperature = np.full(
        len(nodes),
        MATERIAL["ambient_temperature"],
    )

    temperature[:50] = 1200.0

    surface = np.array(
        [node.surface for node in nodes],
        dtype=bool,
    )

    solver = ConvectionSolver(
        density=MATERIAL["density"],
        block_length=CONVECTION["block_length"],
    )

    cooled = solver.block_cooling(
        temperature,
        surface,
        h=CONVECTION["forced"],
        cp=MATERIAL["specific_heat"],
        ambient_temperature=MATERIAL["ambient_temperature"],
        block_time= BLOCK["time_per_block"],
    )

    print(
        "Maximum temperature:",
        np.max(cooled)
    )