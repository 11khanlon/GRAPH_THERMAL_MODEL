"""
goldak.py

Goldak double-ellipsoid heat source model

Implements Equation (19)

Riensche et al. (2023)
Thermal Modeling of Directed Energy Deposition
using Graph Theory
"""

import numpy as np


class GoldakHeatSource:

    def __init__(
        self,
        laser_power,
        thermal_conductivity,
        thermal_diffusivity,
        laser_velocity,
        scaling_factor
    ):

        self.P = laser_power
        self.k = thermal_conductivity
        self.alpha = thermal_diffusivity
        self.v = laser_velocity
        self.C = scaling_factor

    ############################################################
    # Equation (19)
    ############################################################

    def temperature(
        self,
        x,
        y,
        z
    ):
        """
        Goldak temperature field

        Parameters
        ----------
        x,y,z : local coordinates (meters)

        Returns
        -------
        temperature (degC)
        """

        r = np.sqrt(
            x*x +
            y*y +
            z*z
        )

        # avoid singularity at laser center
        r = np.maximum(r, 1e-8)

        numerator = self.C * self.P

        denominator = (
            2 *
            np.pi *
            self.k *
            r
        )

        exponent = np.exp(
            -(self.v / (2*self.alpha))
            * (x + r)
        )

        return numerator / denominator * exponent

    ############################################################
    # Evaluate all nodes
    ############################################################

    def temperature_field(
        self,
        coordinates,
        laser_position
    ):

        temps = np.zeros(len(coordinates))

        lx, ly, lz = laser_position

        for i, point in enumerate(coordinates):

            x = point[0] - lx
            y = point[1] - ly
            z = point[2] - lz

            temps[i] = self.temperature(
                x,
                y,
                z
            )

        return temps

    ############################################################
    # Surface meltpool only
    ############################################################

    def surface_nodes(
        self,
        coordinates,
        laser_position,
        tolerance=0.0002
    ):

        temps = self.temperature_field(
            coordinates,
            laser_position
        )

        mask = np.abs(
            coordinates[:,2] -
            laser_position[2]
        ) < tolerance

        temps[~mask] = 0.0

        return temps

    ############################################################
    # Reheat lower layers
    ############################################################

    def subsurface_nodes(
        self,
        coordinates,
        laser_position,
        cutoff_temperature=326
    ):

        temps = self.temperature_field(
            coordinates,
            laser_position
        )

        temps[temps < cutoff_temperature] = 0.0

        return temps

    ############################################################
    # Approximate meltpool dimensions
    ############################################################

    def meltpool_radius(
        self,
        liquidus_temperature=1630
    ):

        radii = np.linspace(
            0.00001,
            0.003,
            500
        )

        temperatures = []

        for r in radii:

            temperatures.append(
                self.temperature(r,0,0)
            )

        temperatures = np.array(temperatures)

        idx = np.argmin(
            np.abs(
                temperatures -
                liquidus_temperature
            )
        )

        return radii[idx]