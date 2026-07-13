"""
Graph-theory heat conduction solver

Implements the spectral solution presented in

Section 4.3.1.4
Section 4.3.1.6

Paper Equations:
(15) Tc = Φ exp(-alpha g Λ tb) Φᵀ T0
(17) TLc = Φ exp(-alpha g Λ t) Φᵀ Tb

where

Φ  = Laplacian eigenvectors
Λ  = Laplacian eigenvalues
alpha  = thermal diffusivity
g  = gain factor
t  = timestep

This module performs ONLY heat conduction --> It does not apply laser heating or convection
"""

import numpy as np


class ConductionSolver:

    def __init__(self, graph, gain):

        """
        graph : ThermalGraph Graph object after eigensystem() has been computed.
        gain : float, Gain factor g from the paper.
        """

        if graph.eigenvalues is None:
            raise ValueError(
                "Graph eigensystem has not been computed."
            )

        self.graph = graph
        self.gain = gain

        self.lambda_values = graph.eigenvalues
        self.phi = graph.eigenvectors

    # ---------------------------------------------------------

    def propagator(self, alpha, dt):

        """
        Construct:  Φ exp(-alpha g Λ dt) Φᵀ

        Parameters
        alpha : float, Thermal diffusivity (m²/s)
        dt : float, Time step (seconds)

        Return: ndarray, Spectral propagation matrix.
        """

        decay = np.exp(
            -alpha
            * self.gain
            * self.lambda_values
            * dt
        )

        D = np.diag(decay)

        return self.phi @ D @ self.phi.T

    # ---------------------------------------------------------

    def step(self, temperature, alpha, dt):

        """
        Advance one conduction timestep
        Implements Equation (15): Tc = Φ exp(-alphagΛdt) Φᵀ T

        Parameters:
        temperature : ndarray, Current nodal temperatures.
        alpha : float, Thermal diffusivity
        dt : float, Time step

        Returns:
        ndarray --> Updated temperatures after conduction.
        """

        P = self.propagator(alpha, dt)

        return P @ temperature

    # ---------------------------------------------------------

    def block_conduction(
        self,
        temperature,
        alpha,
        block_time
        ):
        

        return self.step(
            temperature,
            alpha,
            block_time
        )

    # ---------------------------------------------------------

    def dwell_conduction(
        self,
        temperature,
        alpha,
        dt=1.0):

        """
        Conduction during dwell time
        Implements Equation (17)

        One call corresponds to one second
        of dwell cooling

        Parameters:
        temperature : ndarray
        alpha : float
        dt : float, Usually 1 second.

        Returns:
        ndarray
        """

        return self.step(
            temperature,
            alpha,
            dt
        )

    # ---------------------------------------------------------

    def propagate(
        self,
        temperature,
        alpha,
        total_time,
        dt):

        """
        Repeatedly apply conduction.
        Useful for arbitrary simulation times.

        Parameters:
        temperature : ndarray
        alpha : float
        total_time : float
        dt : float

        Returns:
        ndarray
        """

        T = temperature.copy()

        nsteps = int(np.ceil(total_time / dt))

        for _ in range(nsteps):

            T = self.step(
                T,
                alpha,
                dt
            )

        return T


# ------------------------------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from graph import ThermalGraph
    from config import GRAPH, MATERIAL, BLOCK

    geom = Geometry("example_part.stl")

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    graph = ThermalGraph(generator.nodes)

    graph.build()

    graph.degree_matrix()

    graph.laplacian()

    graph.eigensystem()

    solver = ConductionSolver(
        graph,
        gain=GRAPH["gain"]
    )

    T = np.full(
        graph.N,
        MATERIAL["ambient_temperature"]
    )

    # Example: heat first 25 nodes
    T[:25] = 2200.0

    T_new = solver.block_conduction(
        temperature=T,
        alpha=MATERIAL["thermal_diffusivity"],
        block_time = BLOCK["time_per_block"]
    )

    print("Maximum temperature:", np.max(T_new))