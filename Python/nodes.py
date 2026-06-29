"""
nodes.py

Creates the point cloud used by the graph-theory solver.

Implements Step 1 from Riensche et al. (2023):

    - Random node generation
    - Uniform node density
    - One point cloud for substrate + deposited part

Each node later becomes one graph vertex.
"""

import numpy as np

from Python.config import GRAPH, MATERIAL


# ---------------------------------------------------------
# Graph Node
# ---------------------------------------------------------

class Node:

    def __init__(self,
                 node_id,
                 position,
                 block_id):

        self.id = node_id

        # Cartesian coordinate
        self.position = np.asarray(position, dtype=float)

        # block containing this node
        self.block_id = block_id

        # becomes True once deposited
        self.active = False

        # ambient temperature initially
        self.temperature = MATERIAL["ambient_temperature"]

        # neighbors filled in graph.py
        self.neighbors = []

        # edge weights
        self.weights = []

    def __repr__(self):

        return (
            f"Node("
            f"id={self.id}, "
            f"block={self.block_id}, "
            f"T={self.temperature:.1f})"
        )


# ---------------------------------------------------------
# Node Generator
# ---------------------------------------------------------

class NodeGenerator:

    """
    Generates random nodes inside every block.

    Paper:
    Section 4.3.1.1

    Random nodes are distributed uniformly inside each
    deposition block.

    Node density controls accuracy.
    """

    def __init__(self, geometry):

        self.geometry = geometry

        self.nodes = []

    # -----------------------------------------------------

    def generate(self):

        node_id = 0

        density = GRAPH["node_density"]  # nodes/mm³

        for block in self.geometry.blocks:

            x0, y0, z0 = block.start
            x1, y1, z1 = block.end

            # -------------------------------
            # Block volume
            # -------------------------------

            volume_mm3 = (
                abs(x1-x0)*1000 *
                abs(y1-y0)*1000 *
                abs(z1-z0)*1000
            )

            n_nodes = max(
                1,
                int(volume_mm3 * density)
            )

            # -------------------------------
            # Uniform random node placement
            # -------------------------------

            for _ in range(n_nodes):

                x = np.random.uniform(x0, x1)
                y = np.random.uniform(y0, y1)
                z = np.random.uniform(z0, z1)

                node = Node(
                    node_id=node_id,
                    position=(x, y, z),
                    block_id=block.id
                )

                self.nodes.append(node)

                block.nodes.append(node)

                node_id += 1

        print("--------------------------------")
        print("Nodes Generated")
        print("--------------------------------")
        print("Total Nodes:", len(self.nodes))
        print()

        return self.nodes

    # -----------------------------------------------------

    def activate_block(self, block):

        """
        Activate every node inside one deposited block.
        """

        for node in block.nodes:

            node.active = True

    # -----------------------------------------------------

    def active_nodes(self):

        return [
            n
            for n in self.nodes
            if n.active
        ]

    # -----------------------------------------------------

    def inactive_nodes(self):

        return [
            n
            for n in self.nodes
            if not n.active
        ]

    # -----------------------------------------------------

    def temperatures(self):

        return np.array([
            n.temperature
            for n in self.nodes
        ])

    # -----------------------------------------------------

    def coordinates(self):

        return np.array([
            n.position
            for n in self.nodes
        ])


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    from Python.geometry import Geometry

    geom = Geometry("example_part.stl")

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    print(generator.nodes[0])