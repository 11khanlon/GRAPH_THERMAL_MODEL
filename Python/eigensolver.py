"""
eigensolver.py

Computes the eigendecomposition of the graph Laplacian.

Implements Equations (7)-(10)

    LΦ = ΦΛ

Temperature evolution is later computed using

    T = Φ exp(-αgΛt) Φᵀ T₀

"""

import numpy as np
from scipy.linalg import eigh


class EigenSolver:

    def __init__(self, laplacian):

        self.L = laplacian

        self.eigenvalues = None
        self.eigenvectors = None

    ###############################################################
    # Compute eigendecomposition
    ###############################################################

    def solve(self):

        print("Computing graph eigensystem...")

        L_dense = self.L.toarray()

        values, vectors = eigh(L_dense)

        self.eigenvalues = values
        self.eigenvectors = vectors

        print("Finished eigendecomposition")
        print(f"{len(values)} eigenvalues computed")

        return values, vectors

    ###############################################################
    # Verify orthogonality
    ###############################################################

    def check_orthogonality(self):

        Phi = self.eigenvectors

        identity = Phi.T @ Phi

        error = np.linalg.norm(
            identity - np.eye(identity.shape[0])
        )

        print(f"Orthogonality error: {error:.3e}")

        return error

    ###############################################################
    # Verify reconstruction
    ###############################################################

    def reconstruction_error(self):

        Phi = self.eigenvectors
        Lambda = np.diag(self.eigenvalues)

        reconstructed = Phi @ Lambda @ Phi.T

        error = np.linalg.norm(
            reconstructed - self.L.toarray()
        )

        print(f"Laplacian reconstruction error: {error:.3e}")

        return error

    ###############################################################
    # Spectral exponential
    ###############################################################

    def spectral_matrix(self,
                        alpha,
                        gain,
                        dt):

        """
        Computes

        Φ exp(-α g Λ dt) Φᵀ

        This matrix advances the temperature one timestep.
        """

        decay = np.exp(
            -alpha * gain * self.eigenvalues * dt
        )

        exponential = np.diag(decay)

        propagator = (
            self.eigenvectors
            @ exponential
            @ self.eigenvectors.T
        )

        return propagator

    ###############################################################
    # Return eigensystem
    ###############################################################

    def get(self):

        return (
            self.eigenvalues,
            self.eigenvectors
        )


if __name__ == "__main__":

    from Python.geometry import ThinWallGeometry
    from node_generator import NodeGenerator
    from Python.graph_builder import GraphBuilder

    geom = ThinWallGeometry()

    nodes = NodeGenerator(geom).generate_nodes()

    graph = GraphBuilder(
        nodes,
        epsilon=0.0025
    )

    _, _, L = graph.build()

    solver = EigenSolver(L)

    values, vectors = solver.solve()

    solver.check_orthogonality()

    solver.reconstruction_error()

    print(values[:10])