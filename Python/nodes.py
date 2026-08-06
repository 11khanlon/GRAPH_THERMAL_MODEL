"""
Step 3 of the graph-theory DED thermal model
Creates the point cloud used by the graph-theory solver.

Implements Step 1 from Riensche et al. (2023):

    - Random node generation
    - Uniform node density
    - One point cloud for substrate + deposited part

Each node later becomes one graph vertex
"""

import numpy as np
from config import GRAPH, MATERIAL, BUILD


#---------- Graph Node -----------
class Node:

    def __init__(self,
                 node_id,
                 position,
                 block_id, 
                 block, 
                 is_substrate=False
                 ):

        self.id = node_id

        # Cartesian coordinate
        self.position = np.asarray(position, dtype = float)

        # block containing this node
        self.block_id = block_id
        self.block = block

        #determine if substrate 
        self.is_substrate = is_substrate

        # becomes True once deposited. Now substrate nodes are active immediately
        # Deposition nodes remain inactive until printed 
        self.active = is_substrate

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
 


# -------- Node Generator ---------

class NodeGenerator:

    """
    Generates random nodes inside every block
    Section 4.3.1.1

    Random nodes are distributed uniformly inside each
    deposition block.

    Node density controls accuracy
    """

    def __init__(self, geometry):

        self.geometry = geometry

        self.nodes = []

    # -----------------------------------------------------
    def generate(self):

        node_id = 0

        node_density = GRAPH["node_density"]  # nodes/mm³

        for block in self.geometry.blocks:

            x0, y0, z0 = block.start
            x1, y1, z1 = block.end

            # Block volume
            volume_mm3 = (
                abs(x1-x0)*1000 *
                abs(y1-y0)*1000 *
                abs(z1-z0)*1000
            )

            expected_nodes = volume_mm3 * node_density

            n_nodes = max(1,int(np.round(expected_nodes)))


            # Uniform random node placement
            for _ in range(n_nodes):

                x = np.random.uniform(x0, x1)
                y = np.random.uniform(y0, y1)
                z = np.random.uniform(z0, z1)

                node = Node(
                    node_id=node_id,
                    position=(x, y, z),
                    block_id=block.id,
                    block=block,
                    is_substrate=block.is_substrate
                )

                self.nodes.append(node)

                block.node_indices.append(node_id)

                block.nodes.append(node)

                node_id += 1

        substrate_nodes = sum(
            node.is_substrate
            for node in self.nodes
        )

        deposition_nodes = len(self.nodes) - substrate_nodes

        print(f"Substrate Nodes : {substrate_nodes}")
        print(f"Deposition Nodes: {deposition_nodes}")
        print(f"Total Nodes     : {len(self.nodes)}")

        self.geometry.nodes = self.nodes

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


    # -----------------------------------------------------
    def validate_nodes(self):
        
        print(f"\nNode Validation")
      
        substrate_nodes = sum(
            node.is_substrate
            for node in self.nodes
        )

        deposition_nodes = len(self.nodes) - substrate_nodes

        active_nodes = sum(
            node.active
            for node in self.nodes
        )

        inactive_nodes = len(self.nodes) - active_nodes

        print(f"Substrate Nodes : {substrate_nodes}")
        print(f"Deposition Nodes: {deposition_nodes}")
        print(f"Total Nodes     : {len(self.nodes)}")

        print()

        print(f"\nActive Nodes    : {active_nodes}")
        print(f"Inactive Nodes  : {inactive_nodes}")


        nodes_per_block = []

        densities = []

        for block in self.geometry.blocks:

            n = len(block.nodes)

            nodes_per_block.append(n)

            x0, y0, z0 = block.start
            x1, y1, z1 = block.end

            volume_mm3 = (
                abs(x1-x0)*1000 *
                abs(y1-y0)*1000 *
                abs(z1-z0)*1000
            )

            densities.append(
                n / volume_mm3
            )

        print(f"\nMinimum Nodes per Block : {min(nodes_per_block)}")
        print(f"Maximum Nodes per Block : {max(nodes_per_block)}")
        print(f"Average Nodes per Block : {np.mean(nodes_per_block):.2f}")


        print(f"\nRequested Density : {GRAPH['node_density']:.3f} nodes/mm³")
        print(f"Average Density   : {np.mean(densities):.3f} nodes/mm³")

        print(f"\nMinimum Density : {min(densities):.3f}")
        print(f"Maximum Density : {max(densities):.3f}")
        print(f"Average Density : {np.mean(densities):.3f}")
        

# --------- Testing ----------
if __name__ == "__main__":

    from geometry import Geometry
    from visualization import  plot_geometry_with_nodes


    stl_file = BUILD["stl_file"]
    
    geom = Geometry(stl_file)

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    print(generator.nodes[0])

    generator.validate_nodes()

    plot_geometry_with_nodes(geom, generator)