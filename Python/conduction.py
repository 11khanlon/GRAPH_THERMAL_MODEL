"""
Step 6: Graph-theory heat conduction solver
Tc = Φ exp(-alpha g Λ tb) Φᵀ T0
"""

import numpy as np

from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

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
        Return: ndarray, Spectral propagation matrix.
        """

        decay = np.exp(-alpha * self.gain * self.lambda_values * dt)

        D = np.diag(decay)

        return self.phi @ D @ self.phi.T

    # ---------------------------------------------------------

    def step(self, temperature, alpha, dt):

        """
        Advance one conduction timestep
        Returns: ndarray --> Updated temperatures after conduction
        """

        P = self.propagator(alpha, dt)

        return P @ temperature

    # ---------------------------------------------------------

    def block_conduction(self, temperature, alpha, block_time):
        
        return self.step(temperature, alpha, block_time)

    # ---------------------------------------------------------

    def dwell_conduction(self,
                         temperature,
                         alpha,
                         dt):

        #Conduction during dwell time --> Implements Equation (17)

        return self.step(temperature,
                         alpha,
                         dt)

    
    # ---------------------------------------------------------

    def propagate(self,
                  temperature,
                  alpha,
                  total_time,
                  block_time):

        
        #Repeatedly apply conduction
      
        T = temperature.copy()

        nsteps = int(np.ceil(total_time / block_time))

        for _ in range(nsteps):

            T = self.step(T, alpha, block_time)

        return T


# ------------------------------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from graph import ThermalGraph
    from config import GRAPH, MATERIAL, BLOCK, BUILD, LASER
    from material import Ti64Material

    stl = BUILD["stl_file"]

    geom = Geometry(stl)

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    graph = ThermalGraph(generator.nodes)

    graph.build()

    graph.degree_matrix()

    graph.laplacian()

    graph.eigensystem()

    solver = ConductionSolver(graph, gain = GRAPH["gain"])

    material = Ti64Material()

    T0 = np.full(graph.N, MATERIAL["ambient_temperature"])

    # Example: heat first 25 nodes
    block = geom.deposition_order()[0]

    for node in block.nodes:
        T0[node.id] = 2200.0

    #alpha = LASER["thermal_diffusivity"]
    alpha = material.layer_diffusivity(T0)

    print("\n----------- Conduction Verification -----------")

    print(
        f"Initial maximum temperature : "
        f"{T0.max():.2f} °C"
    )

    print(
        f"Layer average temperature   : "
        f"{np.mean(T0):.2f} °C"
    )

    print(
        f"Diffusivity                 : "
        f"{alpha:.3e} m²/s"
    )

    # Conduct for one block

    T1 = solver.block_conduction(
        temperature=T0,
        alpha = alpha,
        block_time =BLOCK["time_per_block"]
    )

    print(
        f"Final maximum temperature   : "
        f"{T1.max():.2f} °C"
    )

    print(
        f"Final minimum temperature   : "
        f"{T1.min():.2f} °C"
    )

    print("\nTemperature statistics")
    
    print("T0:")
    print("  max :", T0.max())
    print("  min :", T0.min())
    print("  mean:", T0.mean())

    print("\nT1:")
    print("  max :", T1.max())
    print("  min :", T1.min())
    print("  mean:", T1.mean())

    difference = T1 - T0

    print("\n Temperature Change:")
    print("Maximum temperature change:",
      T1.max() - T0.max())

    print("Largest local increase:",
        difference.max())

    print("Largest local decrease:",
        difference.min())

    print("Largest absolute local change:",
        np.max(np.abs(difference)))

    print(f"\n -----------------------------------------------")

    print("Temperature sum:", np.max(np.abs(T1 - T0)))

    print("Final energy proxy:", np.sum(T1))

    print(f"\n ----------------------------------")
    print("Smallest eigenvalue:", solver.lambda_values[0])
    print("Largest eigenvalue :", solver.lambda_values[-1])

    print("Decay factors:")
    print(
        np.exp(
            -alpha *
            solver.gain *
            solver.lambda_values[[0, 1, 2, -3, -2, -1]] *
            BLOCK["time_per_block"]
        )
    )

