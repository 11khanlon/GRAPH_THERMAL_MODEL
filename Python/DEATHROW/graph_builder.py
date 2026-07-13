"""
Constructs the weighted graph used for graph-theory thermal diffusion

Implements Equations (11)-(14) 
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix, csr_matrix, diags


class GraphBuilder:

    def __init__(self, nodes, epsilon):

        """
        nodes : ndarray (N x 3) Cartesian node coordinates
        epsilon : float Neighborhood radius (meters)
        """

        self.nodes = nodes
        self.epsilon = epsilon

        self.N = len(nodes)

        self.adjacency = None
        self.degree = None
        self.laplacian = None


    #---------- Build Graph ----------------

    def build(self):

        tree = cKDTree(self.nodes)

        adjacency = lil_matrix((self.N, self.N))

        
        # First gather all neighbor distances
        pair_distances = []

        neighbor_lists = []

        for i in range(self.N):

            neighbors = tree.query_ball_point(
                self.nodes[i],
                self.epsilon
            )

            if i in neighbors:
                neighbors.remove(i)

            neighbor_lists.append(neighbors)

            for j in neighbors:

                if j > i:

                    d = np.linalg.norm(
                        self.nodes[i] - self.nodes[j]
                    )

                    pair_distances.append(d)

        pair_distances = np.array(pair_distances)

        sigma = np.std(pair_distances)

        print(f"Neighborhood sigma = {sigma:.6e}")

    
        # Build weighted adjacency matrix

        for i in range(self.N):

            for j in neighbor_lists[i]:

                d = np.linalg.norm(
                    self.nodes[i] - self.nodes[j]
                )

                weight = np.exp(-(d**2)/(sigma**2))

                adjacency[i, j] = weight
                adjacency[j, i] = weight

        adjacency = adjacency.tocsr()

       
        # Degree Matrix

        degree_values = np.array(
            adjacency.sum(axis=1)
        ).flatten()

        degree = diags(degree_values)

    
        # Laplacian

        laplacian = degree - adjacency

        self.adjacency = adjacency
        self.degree = degree
        self.laplacian = laplacian

        print("Graph construction complete")
        print(f"Nodes : {self.N}")
        print(f"Edges : {adjacency.nnz//2}")

        return adjacency, degree, laplacian


    # Average neighbors

    def average_neighbors(self):

        if self.adjacency is None:
            return 0

        degrees = np.array(
            self.adjacency.sum(axis=1)
        ).flatten()

        return np.mean(degrees > 0)


    # Connectivity check

    def connectivity(self):

        if self.adjacency is None:
            return

        degree = np.array(
            self.adjacency.sum(axis=1)
        ).flatten()

        isolated = np.sum(degree == 0)

        print(f"Isolated nodes: {isolated}")

   
    # Return matrices

    def get_graph(self):

        return (
            self.adjacency,
            self.degree,
            self.laplacian
        )


if __name__ == "__main__":

    from node_generator import NodeGenerator
    from Python.geometry import ThinWallGeometry

    geom = ThinWallGeometry()

    nodes = NodeGenerator(geom).generate_nodes()

    builder = GraphBuilder(
        nodes,
        epsilon=0.0025
    )

    A, D, L = builder.build()

    builder.connectivity()

    print(A.shape)
    print(L.shape)