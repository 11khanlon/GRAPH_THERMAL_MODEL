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


from config import BUILD, BLOCK, WAAM

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
        is_substrate=False):

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
            "back": False}

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

    def __init__(self, stl_file, substrate_material, wire_material):

        self.substrate_material = substrate_material
        self.wire_material = wire_material

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


        # Convert STL from mm -> m
        self.mesh.apply_scale(1e-3)

 
        # SUBSTRATE PARAMETERS
        self.substrate_length = BUILD["substrate_length"]
        self.substrate_width = BUILD["substrate_width"]
        self.substrate_height = BUILD["substrate_height"]

        self.substrate_blocks_x = BUILD["substrate_blocks_x"]
        self.substrate_blocks_y = BUILD["substrate_blocks_y"]
        self.substrate_blocks_z = BUILD["substrate_blocks_z"]

         #--------------------
        # IMPORTANT: initialize block storage
        self.blocks = []
        self.next_block_id = 0

        #--------------------
        # ORIGINAL STL BOUNDS
        bounds = self.mesh.bounds

        xmin, ymin, zmin = bounds[0]
        xmax, ymax, zmax = bounds[1]

        stl_length = xmax - xmin
        stl_width = ymax - ymin
        stl_height = zmax - zmin

         #--------------------
        # CENTER STL ON SUBSTRATE
        target_xmin = (self.substrate_length - stl_length) / 2

        target_ymin = (self.substrate_width - stl_width) / 2
        target_zmin = self.substrate_height

        dx = target_xmin - xmin
        dy = target_ymin - ymin
        dz = target_zmin - zmin

        self.mesh.apply_translation([dx, dy, dz])

        #--------------------
        # UPDATED STL BOUNDS
        self.bounds = self.mesh.bounds

        xmin, ymin, zmin = self.bounds[0]
        xmax, ymax, zmax = self.bounds[1]

        self.length = xmax - xmin
        self.width = ymax - ymin
        self.height = zmax - zmin

        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        self.zmin = zmin
        self.zmax = zmax

        #-------------------------
        # SUBSTRATE BOUNDS
        self.substrate_xmin = 0.0
        self.substrate_xmax = self.substrate_length

        self.substrate_ymin = 0.0
        self.substrate_ymax = self.substrate_width

        self.substrate_bottom = 0.0
        self.substrate_top = self.substrate_height


    def generate_substrate_nodes(self, substrate_coords):

        for xyz in substrate_coords:
            self.coords.append(xyz)
            self.material_ids.append(0)  # Al6061

    def generate_bead_nodes(self, bead_coords):

        for xyz in bead_coords:
            self.coords.append(xyz)
            self.material_ids.append(1)  # Al5183
            
    # -----------------------------------------
    def build_substrate(self):

        nx = self.substrate_blocks_x
        ny = self.substrate_blocks_y
        nz = self.substrate_blocks_z

        block_length = self.substrate_length / nx
        block_width  = self.substrate_width / ny
        block_height = self.substrate_height / nz

        xmin = 0.0
        xmax = self.substrate_length

        ymin = 0.0
        ymax = self.substrate_width

        zmin = 0.0
        zmax = self.substrate_height


        for k in range(nz):

            z0 = zmin + k * block_height
            z1 = min(z0 + block_height, zmax)

            for j in range(ny):

                y0 = ymin + j * block_width
                y1 = min(y0 + block_width, ymax)

                for i in range(nx):

                    x0 = xmin + i * block_length
                    x1 = min(x0 + block_length, xmax)

                    start = (x0, y0, z0)
                    end   = (x1, y1, z1)

                    self.blocks.append(
                        Block(
                            self.next_block_id,
                            k - nz,
                            start,
                            end,
                            is_substrate=True
                        )
                    )

                    self.next_block_id += 1

    #--------------------------------   
    #Create deposition blocks and area 

    def build_deposition(self):

        
        block_height = BLOCK["height"]
        total_layers = int(np.ceil(self.height / BLOCK["height"]))
        z = self.zmin

        blocks_per_layer = BUILD["blocks_per_layer"]
        block_length = self.length / blocks_per_layer
        time_per_block = block_length / WAAM["scan_speed"]

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
        print(f"Total blocks:", len(self.blocks))

    
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

    import os
    import sys

    # Get the parent "Python" folder
    python_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    # Add it to Python's import search path
    if python_folder not in sys.path:
        sys.path.insert(0, python_folder)
        
    from Visualization.geometry_visualization import plot_geometry, check_geometry_locations
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


    