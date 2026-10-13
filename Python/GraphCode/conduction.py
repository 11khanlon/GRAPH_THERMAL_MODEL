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
            raise ValueError("Graph eigensystem has not been computed.")

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

    
class HeterogeneousConductionSolver:

    def __init__(self, graph, gain=1.0):

        self.graph = graph
        self.nodes = graph.nodes
        self.gain = gain

    def step(self, temperature, dt, active_mask, volumes, edge_areas):

        temperature = np.asarray(temperature, dtype=float)
        active_mask = np.asarray(active_mask, dtype=bool)
        volumes = np.asarray(volumes, dtype=float)

        # 1. Temperature-dependent material properties
        rho = np.array([
            node.material.rho
            for node in self.nodes
        ], dtype=float)

        cp = np.array([
            node.material.specific_heat(temperature[i])
            for i, node in enumerate(self.nodes)
        ], dtype=float)

        k = np.array([
            node.material.conductivity(temperature[i])
            for i, node in enumerate(self.nodes)
        ], dtype=float)

        # 2. Active nodes only
        active_ids = np.flatnonzero(active_mask)
        local_index = np.full(len(self.nodes), -1, dtype=int)
        local_index[active_ids] = np.arange(len(active_ids))

        # 3. Thermal capacitance
        C = rho[active_ids] * cp[active_ids] * volumes[active_ids]

        # 4. Construct conductance-weighted Laplacian
        adjacency = self.graph.adjacency.tocoo()

        rows = []
        cols = []
        values = []

        for i, j in zip(adjacency.row, adjacency.col):

            # Process each undirected edge once
            if i >= j:
                continue

            # No conduction through nonexistent material
            if not (active_mask[i] and active_mask[j]):
                continue

            distance = np.linalg.norm(
                np.asarray(self.nodes[i].position) -
                np.asarray(self.nodes[j].position)
            )

            if distance <= 0:
                raise ValueError("Connected nodes have zero separation.")

            area = edge_areas(i, j)

            conductance = (
                self.gain
                * area / distance
                * (2 * k[i] * k[j] / (k[i] + k[j]))
            )

            a = local_index[i]
            b = local_index[j]

            rows.extend([a, a, b, b])
            cols.extend([a, b, a, b])
            values.extend([
                conductance, -conductance,
                -conductance, conductance
            ])

        n = len(active_ids)

        L_G = coo_matrix(
            (values, (rows, cols)),
            shape=(n, n)
        ).tocsr()

        # 5. dT/dt = -C^-1 L_G T
        A = -diags(1.0 / C) @ L_G

        # 6. Advance temperatures
        updated = temperature.copy()

        updated[active_ids] = expm_multiply(
            A * dt,
            temperature[active_ids]
        )

        return updated


# ------------------------------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from graph import ThermalGraph
    from config import GRAPH, MATERIAL, BLOCK, BUILD, WAAM
    from material import Al5183Material, Al6061Material

    stl_file = BUILD["stl_file"]
    substrate_material = Al6061Material()
    wire_material = Al5183Material()
    geom = Geometry(stl_file, substrate_material=substrate_material, wire_material=wire_material)

    geom.build_blocks()
    generator = NodeGenerator(geom)
    generator.generate()
    graph = ThermalGraph(generator.nodes)
    graph.build()
    graph.degree_matrix()
    graph.laplacian()
    graph.eigensystem()
    solver = ConductionSolver(graph, gain = GRAPH["gain"])


    T0 = np.full(graph.N, MATERIAL["ambient_temperature"])

    block = geom.deposition_order()[0]

    # Activate the newly deposited block
    for node in block.nodes:
        node.active = True

    block_node_ids = np.array([
        node.id
        for node in block.nodes
    ], dtype=int)

    if block_node_ids.size == 0:
        raise ValueError(
            "Selected deposition block contains no nodes. "
            "Check node generation and block-node assignment.")

    # Apply test heat
    T0[block_node_ids] = 1500.0

    T_average = np.mean(T0[block_node_ids])

    alpha = block.material.diffusivity(T_average)

    print(f"Current block average temperature : " f"{T_average:.2f} °C")
    print(f"Thermal diffusivity: " f"{alpha:.3e} m²/s")

    print("\n----------- Conduction Verification -----------")
    print(f"Initial maximum temperature : "f"{T0.max():.2f} °C")

    print(f"Layer average temperature   : "
        f"{np.mean(T0):.2f} °C")

    print(f"Diffusivity: "
        f"{alpha:.3e} m²/s")

    # Conduct for one block
    T1 = solver.block_conduction(
        temperature=T0,
        alpha = alpha,
        block_time =BLOCK["time_per_block"])

    print(f"Final maximum temperature: "
        f"{T1.max():.2f} °C")

    print(f"Final minimum temperature: "
        f"{T1.min():.2f} °C")

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


    alpha_nodes = np.array([
        node.material.diffusivity(T0[node.id])
        for node in graph.nodes
    ])

    print("\n--- Material Diffusivity Verification ---")

    for material_name in ["Al6061", "Al5183"]:

        ids = np.array([
            node.id
            for node in generator.nodes
            if node.material.name == material_name
        ])

        print(f"{material_name}:")
        print(f"  Average alpha: {np.mean(alpha_nodes[ids]):.3e} m²/s")
        print(f"  Minimum alpha: {np.min(alpha_nodes[ids]):.3e} m²/s")
        print(f"  Maximum alpha: {np.max(alpha_nodes[ids]):.3e} m²/s")

