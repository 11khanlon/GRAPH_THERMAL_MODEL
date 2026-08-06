"""
Step 4

Constructs the graph used for thermal diffusion.
Implements Section 4.3.1.2 and creates:

A = adjacency matrix
H = degree matrix
L = graph Laplacian

computes
Lφ = φΛ
"""

import numpy as np

from scipy.spatial import cKDTree
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
import time 


from config import GRAPH


class ThermalGraph:

    def __init__(self, nodes):

        self.nodes = nodes

        self.N = len(nodes)

        self.A = None
        self.H = None
        self.L = None

        self.eigenvalues = None
        self.eigenvectors = None

    # --------------------------------------------------------

    def coordinates(self):

        return np.array(
            [node.position for node in self.nodes]
        )

    # --------------------------------------------------------

    def build(self):

        """
        Construct graph using neighborhood radius ε
        Paper Eq. (11)
        """
        t0 = time.perf_counter() 

        coords = self.coordinates()

        tree = cKDTree(coords)

        epsilon = GRAPH["epsilon"]

        # adjacency matrix
        self.A = lil_matrix((self.N, self.N))

        # ----------------------------------------------------
        # Estimate σ (standard deviation of pairwise distances)

        sample = tree.query(coords, k=8)[0][:, 1:]

        sigma = np.std(sample)

        print("Neighborhood radius =", epsilon)
        print("Gaussian sigma =", sigma)

        # ----------------------------------------------------

        for i in range(self.N):

            neighbors = tree.query_ball_point(
                coords[i],
                epsilon
            )

            for j in neighbors:

                if i == j:
                    continue

                distance = np.linalg.norm(
                    coords[i] - coords[j]
                )

                # Equation (11)

                weight = np.exp(
                    -(distance ** 2) /
                    (sigma ** 2)
                )

                self.A[i, j] = weight
                self.A[j, i] = weight

                self.nodes[i].neighbors.append(j)
                self.nodes[i].weights.append(weight)

        #convert from construction format to computation format 
        self.A = self.A.tocsr()

        print(
            f"\n Graph assembly time: "
            f"{time.perf_counter()-t0:.2f} s"
        )

        print("Adjacency matrix complete.")

    # --------------------------------------------------------
    #Equation (13)
    def degree_matrix(self):

        degrees = np.asarray(self.A.sum(axis=1)).flatten()

        self.H = diags(degrees)

        print(f"\nDegree matric verification: ")
        print(self.H.diagonal()[:10])

        return self.H

    # --------------------------------------------------------

    def laplacian(self):

        """
        L = H - A
        Equation (14)
        """

        if self.H is None:

            self.degree_matrix()

        self.L = self.H - self.A

        return self.L

    # --------------------------------------------------------

    def eigensystem(self):

        """
        Compute Lφ = φΛ
        Used later for the spectral heat equation
        """
        t0 = time.perf_counter()

        if self.L is None:

            self.laplacian()

        print("Computing eigenvectors...")

        L_dense = self.L.toarray()

        eigenvalues, eigenvectors = eigh(L_dense)

        self.eigenvalues = eigenvalues

        self.eigenvectors = eigenvectors

        print("\nCheck eigensystem assembly time: ")
        print(f"Eigen solve: " f"{time.perf_counter()-t0:.2f} s")

        print("\nEigen decomposition complete.")

        return eigenvalues, eigenvectors
    
    
    # --------------------------------------------------------

    def summary(self):

        print("\n----------- Graph Summary -----------")

        print("Nodes:", self.N)

        edges = self.A.nnz // 2

        print("Edges:", edges)

        avg_degree = np.mean(
        self.A.getnnz(axis=1))

        degrees = self.A.getnnz(axis=1)
        isolated = np.sum(degrees == 0)

        print(f"\nNeighbor information:")
        print("Average neighbors:", round(avg_degree, 2))
        print("Minimum neighbors :", degrees.min())
        print("Maximum neighbors :", degrees.max())
        print("Average neighbors :", degrees.mean())
        print("Median neighbors  :", np.median(degrees))

        print("\n Verify substrate is connected to deposition connections:")
        substrate_to_part = 0

        for i in range(self.N):

            for j in self.nodes[i].neighbors:

                if self.nodes[i].is_substrate != self.nodes[j].is_substrate:

                    substrate_to_part += 1

        print()
        print("Substrate-Deposition edges:",
            substrate_to_part // 2)

        print(f"\n Check connected components")
        n_components, labels = connected_components(self.A)
        print("Connected components:", n_components)

        print(f"\n Check Symmetry:")
        difference = self.A - self.A.T
        print("Symmetric:",
            difference.nnz == 0)

        print(f"\n Laplacian verification:")
        row_sum = np.abs(self.L.sum(axis=1))
        print("Maximum row sum:", row_sum.max()) 

        print(f"\nSmallest eigenvalues")
        print(self.eigenvalues[:10])
        print(f"Largest eigenvalue")
        print(self.eigenvalues[-1])

        print("\nIsolated nodes:", isolated)
        print("Minimum weight:",self.A.data.min())
        print("Maximum weight:", self.A.data.max())

    

        print("-------------------------------------")

    # --------------------------------------------------------

    def active_subgraph(self):

        """
        Returns indices of active nodes
        Used during deposition
        """

        active = []

        for i, node in enumerate(self.nodes):

            if node.active:

                active.append(i)

        return np.array(active)


# ------------------------------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from config import BUILD
    

    stl_file = BUILD["stl_file"]
    
    geom = Geometry(stl_file)

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    graph = ThermalGraph(generator.nodes)

    graph.build()

    graph.degree_matrix()

    graph.laplacian()

    graph.eigensystem()

    graph.summary()