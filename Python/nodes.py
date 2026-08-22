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

        # Faces currently exposed to the environment, Updated dynamically as deposition progresses.
        self.exposed_faces = set()

        # Convection boundary-condition classification
        self.forced_surface = False
        self.free_surface = False

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

    #-------------------------------------------------- 
    
def node_on_face(node, block, face, thickness):
    """
    Determine whether a node lies within the surface
    region associated with a particular block face.

    The thickness is a numerical surface-band parameter.
    It is NOT a threshold specified by Riensche et al.
    """

    x, y, z = node.position

    x0, y0, z0 = block.start
    x1, y1, z1 = block.end

    if face == "left":
        return abs(x - x0) <= thickness

    elif face == "right":
        return abs(x1 - x) <= thickness

    elif face == "front":
        return abs(y - y0) <= thickness

    elif face == "back":
        return abs(y1 - y) <= thickness

    elif face == "bottom":
        return abs(z - z0) <= thickness

    elif face == "top":
        return abs(z1 - z) <= thickness

    return False


#%%
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
    def update_node_exposure(self, surface_thickness):

        # Clear previous exposure
        for node in self.nodes:
            node.exposed_faces.clear()

        # Examine active blocks
        for block in self.geometry.blocks:

            if not block.active:
                continue

            for face, exposed in block.exposed_faces.items():

                if not exposed:
                    continue

                for node in block.nodes:

                    if node_on_face(
                        node,
                        block,
                        face,
                        surface_thickness
                    ):
                        node.exposed_faces.add(face)

    # -----------------------------------------------------
    def update_surface_types(self):
        """
        Classify exposed nodes according to their convection boundary condition.

        Forced convection:
            Exposed surfaces of deposited material
            Exposed top surface of substrate

        Free convection:
            Exposed sides and bottom of substrate
        """

        for node in self.nodes:

            # Reset previous classification
            node.forced_surface = False
            node.free_surface = False

            # Ignore inactive deposition nodes
            if not node.active:
                continue

            # No exposed surface
            if not node.exposed_faces:
                continue

            # ---------------------------------------------
            # SUBSTRATE

            if node.is_substrate:

                # Substrate top -> forced convection
                if "top" in node.exposed_faces:
                    node.forced_surface = True
                    node.free_surface = False

                # Substrate sides/bottom -> free convection
                elif any(
                    face in node.exposed_faces
                    for face in [
                        "left",
                        "right",
                        "front",
                        "back",
                        "bottom"
                    ]
                ):

                    node.free_surface = True

            # DEPOSITION
            else:
                # Exposed deposition surfaces -> forced convection
                node.forced_surface = True

 
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


# -----------------------------------------------------
def print_surface_statistics(nodes):
    """
    Print statistics for currently exposed nodes.
    """

    surface_nodes = [
        node
        for node in nodes
        if node.exposed_faces
    ]

    print("\n------ SURFACE NODE STATISTICS ------")

    print(f"Total nodes      : {len(nodes)}")

    print(f"Surface nodes    : {len(surface_nodes)}")

    if len(nodes) > 0:

        percentage = (100 * len(surface_nodes) / len(nodes))

        print(f"Surface fraction : {percentage:.2f}%")

    face_counts = {
        "top": 0,
        "bottom": 0,
        "left": 0,
        "right": 0,
        "front": 0,
        "back": 0
    }

    for node in surface_nodes:

        for face in node.exposed_faces:

            face_counts[face] += 1

    for face, count in face_counts.items():

        print(
            f"{face:>7}: {count}"
        )

    print("-------------------------------------")
        

# --------- Testing ----------
if __name__ == "__main__":

    from geometry import Geometry
    from visualization import  plot_geometry_with_nodes
    from face import update_exposed_faces
    from config import BLOCK

    stl_file = BUILD["stl_file"]
    
    geom = Geometry(stl_file)

    geom.build_blocks()

    generator = NodeGenerator(geom)

    nodes = generator.generate()

    update_exposed_faces(geom)

    surface_thickness = BLOCK["height"]/2
    generator.update_node_exposure(surface_thickness)

    print(generator.nodes[0])

    generator.validate_nodes()

    #plot_geometry_with_nodes(geom, generator)

    print_surface_statistics(nodes)