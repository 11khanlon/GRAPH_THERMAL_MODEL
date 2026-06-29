"""
material.py

Temperature-dependent material properties for Ti-6Al-4V.

Implements Section 4.3.4.3 of

Riensche et al.
Thermal Modeling of Directed Energy Deposition
using Graph Theory (2023)
"""

import numpy as np


class Ti64Material:

    ###############################################################
    # Constructor
    ###############################################################

    def __init__(self):

        # Density (constant)
        self.rho = 4430.0       # kg/m^3

        ###########################################################
        # Table 6 from the paper
        #
        # Replace these values with the exact Table 6 values if
        # you have them available.
        ###########################################################

        self.temperature_table = np.array([
            25,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600
        ])

        # Thermal conductivity (W/m-K)
        self.k_table = np.array([
            6.7,
            7.0,
            7.5,
            8.0,
            8.7,
            9.3,
            10.0,
            10.7,
            11.5,
            12.3,
            13.0,
            13.8,
            14.5,
            15.0,
            15.6,
            16.0,
            16.5
        ])

        # Specific heat (J/kg-K)

        self.cp_table = np.array([
            560,
            600,
            640,
            675,
            710,
            740,
            770,
            790,
            810,
            825,
            840,
            850,
            860,
            865,
            870,
            875,
            880
        ])

    ###############################################################
    # Thermal conductivity
    ###############################################################

    def conductivity(self, T):

        return np.interp(
            T,
            self.temperature_table,
            self.k_table
        )

    ###############################################################
    # Specific heat
    ###############################################################

    def specific_heat(self, T):

        return np.interp(
            T,
            self.temperature_table,
            self.cp_table
        )

    ###############################################################
    # Thermal diffusivity
    ###############################################################

    def diffusivity(self, T):

        k = self.conductivity(T)

        cp = self.specific_heat(T)

        return k / (self.rho * cp)

    ###############################################################
    # Paper linear approximation
    ###############################################################

    def diffusivity_linear(self, T):

        """
        Section 4.3.4.3

        α(T) = 0.0042 T + 2.612

        Returns mm²/s
        """

        return (
            0.0042 * T +
            2.612
        )

    ###############################################################
    # Convert mm²/s to m²/s
    ###############################################################

    def diffusivity_linear_si(self, T):

        return (
            self.diffusivity_linear(T)
            * 1e-6
        )

    ###############################################################
    # Average layer diffusivity
    ###############################################################

    def layer_diffusivity(
        self,
        temperatures
    ):

        Tavg = np.mean(temperatures)

        return self.diffusivity_linear_si(Tavg)

    ###############################################################
    # Print material summary
    ###############################################################

    def summary(self):

        print("Material : Ti-6Al-4V")

        print(f"Density : {self.rho} kg/m³")

        print(
            f"k(25°C) = {self.conductivity(25):.2f} W/m-K"
        )

        print(
            f"Cp(25°C) = {self.specific_heat(25):.1f} J/kg-K"
        )

        print(
            f"α(25°C) = {self.diffusivity_linear_si(25):.3e} m²/s"
        )


if __name__ == "__main__":

    material = Ti64Material()

    material.summary()

    temperatures = np.linspace(
        25,
        1600,
        10
    )

    print()

    print("Temperature-dependent properties")

    for T in temperatures:

        print(
            f"{T:7.1f} °C | "
            f"k={material.conductivity(T):6.2f} | "
            f"Cp={material.specific_heat(T):7.1f} | "
            f"α={material.diffusivity_linear_si(T):.3e}"
        )