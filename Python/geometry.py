"""
Step 2 of the graph-theory DED thermal model
Create blocks that is later populated with random nodes 

This module:

1. Loads an STL file
2. Computes its bounding box
3. Divides the build into deposition layers
4. Divides every hatch into five deposition blocks
5. Stores deposition order

"""

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os 
import time


from config import BUILD, BLOCK

# -----------------------------------------
class Block:

    """
    Represents one discretized block
    A block may belong to either:
    • substrate
    • deposited material
    """

    def __init__(
        self,
        block_id,
        layer,
        start,
        end,
        is_substrate=False
    ):

        self.id = block_id
        self.layer = layer
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)

        self.center = (self.start + self.end) / 2

        self.is_substrate = is_substrate
        self.active = is_substrate

        self.nodes = []
        self.node_indices = []

        self.exposed_faces = {
            "top": False,
            "bottom": False,
            "left": False,
            "right": False,
            "front": False,
            "back": False
        }

    def __repr__(self):

        region = "Substrate" if self.is_substrate else "Deposition"

        return (
            f"Block("
            f"id={self.id}, "
            f"{region}, "
            f"layer={self.layer})"
        )

# -----------------------------------------
class Geometry:

    def __init__(self, stl_file):

        # Load deposited part
        self.mesh = trimesh.load_mesh(stl_file)

        # -----------------------------------------
        #Check if Trimesh import works 

        print("Loading STL...")

        if not os.path.exists(stl_file):
            raise FileNotFoundError(
                f"STL file not found:\n{stl_file}"
            )

        start = time.perf_counter()

        try:
            self.mesh = trimesh.load_mesh(stl_file)

        except Exception as e:
            raise RuntimeError(
                f"Unable to load STL:\n{e}"
            )

        elapsed = time.perf_counter() - start

        print(f"STL loaded in {elapsed:.2f} seconds. Success!")

        #---------------------------------
        #Import mesh geometry

        self.mesh.apply_scale(1e-3)

        # Dimensions obtained directly from STL
        self.bounds = self.mesh.bounds
        xmin, ymin, zmin = self.bounds[0]
        xmax, ymax, zmax = self.bounds[1]

        self.length = xmax - xmin
        self.width = ymax - ymin
        self.height = zmax - zmin

        # Bounding box
        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        self.zmin = zmin
        self.zmax = zmax

        # Substrate dimensions
        self.substrate_length = BUILD["substrate_length"]
        self.substrate_width = BUILD["substrate_width"]
        self.substrate_height = BUILD["substrate_height"]

        
        self.blocks = []
        self.next_block_id = 0


    # -----------------------------------------
    def build_substrate(self):

        """
        Discretize the entire substrate into blocks
        The top of the substrate is flush with the bottom of the deposited part.
        """

        #Determine number of blocks in the x,y,z
        nx = 10
        ny = 9
        nz = 28
    
        block_length = self.substrate_length / nx
        block_width  = self.substrate_width / ny
        block_height = self.substrate_height / nz

        # Substrate bounds
        xmin = self.xmin - (self.substrate_length - self.length)/2
        xmax = xmin + self.substrate_length

        ymin = self.ymin - (self.substrate_width - self.width)/2
        ymax = ymin + self.substrate_width

        substrate_bottom = self.zmin - self.substrate_height
        substrate_top = self.zmin

     
        #Create substrate blocks
        for k in range(nz):

            z0 = self.zmin - (nz - k) * block_height
            z1 = z0 + block_height

            for j in range(ny):

                y0 = ymin + j * block_width
                y1 = min(y0 + block_width, ymax)

                for i in range(nx):

                    x0 = xmin + i * block_length
                    x1 = min(x0 + block_length, xmax)

                    start = (x0, y0, z0)
                    end = (x1, y1, z1)

                    self.blocks.append(
                        Block(
                            self.next_block_id,
                            k - nz,
                            start,
                            end,
                            is_substrate=True,
                        )
                    )

                    self.next_block_id += 1

    #--------------------------------   
    #Create deposition blocks and area 

    def build_deposition(self):

        block_length = BLOCK["length"]
        block_height = BLOCK["height"]
        total_layers = int(np.ceil(self.height / BLOCK["height"]))
        z = self.zmin

        blocks_per_layer = BUILD["blocks_per_layer"]

        # Center the deposition on the substrate
        deposit_xmin = self.xmin
        deposit_ymin = self.ymin

        for layer in range(total_layers):

            z0 = self.zmin + layer * block_height
            z1 = z0 + block_height

            for i in range(blocks_per_layer):

                x0 = deposit_xmin + i * block_length
                x1 = x0 + block_length

                start = (x0, deposit_ymin, z0)
                end = (x1, deposit_ymin + BLOCK["width"], z1)

                self.blocks.append(

                    Block(
                        self.next_block_id,
                        layer,
                        start,
                        end,
                        is_substrate=False
                    )
                )

                self.next_block_id += 1


    # -----------------------------------------
    def build_blocks(self):

        self.build_substrate()

        self.build_deposition()

    # ---------------------------------------------
    def deposition_order(self):
        """
        Return deposited blocks in laser scan order

        Even layers: left -> right
        Odd layers: right -> left
        """

        ordered_blocks = []

        total_layers = int(np.ceil(self.height / BLOCK["height"]))

        for layer in range(total_layers):

            layer_blocks = [
                block for block in self.blocks
                if (not block.is_substrate
                    and block.layer == layer
                )
            ]

            # Sort by x coordinate
            layer_blocks.sort(
                key=lambda block: block.start[0]
            )

            # Reverse every other layer
            if layer % 2 == 1:

                layer_blocks.reverse()

            ordered_blocks.extend(layer_blocks)

        return ordered_blocks
    
   
    # --------------------------------------------------------------
    def validate_geometry(self):


        print("------- Geometry Validation --------")
      
        # Part dimensions
        print("\nDeposited Part")

        print(f"Length : {self.length:.4f} m")
        print(f"Width  : {self.width:.4f} m")
        print(f"Height : {self.height:.4f} m")

        print("\nSubstrate")

        print(f"Length : {self.substrate_length:.4f} m")
        print(f"Width  : {self.substrate_width:.4f} m")
        print(f"Height : {self.substrate_height:.4f} m")

       
        # Count blocks
        substrate_blocks = sum(
            block.is_substrate
            for block in self.blocks
        )

        deposition_blocks = sum(
            not block.is_substrate
            for block in self.blocks
        )


        print(f"\nSubstrate blocks :", substrate_blocks)
        print(f"Deposition blocks:", deposition_blocks)
        print(f"Total blocks     :", len(self.blocks))

    
        # Check interface
        substrate_top = max(
            block.end[2]
            for block in self.blocks
            if block.is_substrate)

        deposition_bottom = min(
            block.start[2]
            for block in self.blocks
            if not block.is_substrate)

        gap = deposition_bottom - substrate_top
        ids = [ block.id for block in geom.blocks]

        active = sum(
        block.active
        for block in geom.blocks)

        print(f"\nActive blocks:", active)


        layers = sorted(set(block.layer for block in geom.blocks))
        print(f"\nTotal Layers:", layers)
     

        print(f"\nSubstrate top     : {substrate_top:.6f} m")
        print(f"Deposition bottom : {deposition_bottom:.6f} m")
        print(f"Gap               : {gap:.6e} m")

        if abs(gap) < 1e-9:
            print("PASS: Substrate and deposition touch.")

        else:
            print("WARNING: Gap detected.")

        

# -------------------------------------

if __name__ == "__main__":

    from visualization import plot_geometry, check_geometry_locations, plot_geometry_with_nodes
    from face import update_exposed_faces, print_exposed_faces, print_exposure_statistics

    stl_file = BUILD["stl_file"]
    
    geom = Geometry(stl_file)

    geom.build_blocks()

    print(geom.blocks[0])

    print(geom.blocks[-1])

    plot_geometry(geom)

    geom.validate_geometry()

    update_exposed_faces(geom)

    #print_exposed_faces(geom)

    print_exposure_statistics(geom)


    