"""
Step 6: 
Graph-theory heat conduction solver
Tc = Φ exp(-alpha g Λ tb) Φᵀ T0
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
        Returns: ndarray --> Updated temperatures after conduction
        """

        P = self.propagator(alpha, dt)

        return P @ temperature

    # ---------------------------------------------------------

    def block_conduction(self, 
                        temperature, 
                        alpha,
                        block_time):
        
        return self.step(temperature, 
                        alpha, 
                        block_time)

    # ---------------------------------------------------------

    def dwell_conduction(self,
                         temperature,
                         alpha,
                         dt=1.0):

        #Conduction during dwell time --> Implements Equation (17)

        return self.step(temperature,
                         alpha,
                         dt)

    # ---------------------------------------------------------

    def propagate(self,
                  temperature,
                  alpha,
                  total_time,
                  dt):

        
        #Repeatedly apply conduction
      
        T = temperature.copy()

        nsteps = int(np.ceil(total_time / dt))

        for _ in range(nsteps):

            T = self.step(T, alpha, dt)

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
        gain = GRAPH["gain"]
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