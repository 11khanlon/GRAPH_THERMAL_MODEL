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

from Python.config import BUILD, BLOCK


class Block:

    """
    Represents one deposition block.
    Every block is activated once during the simulation.
    """

    def __init__(self,
                 block_id,
                 layer,
                 start,
                 end):

        self.id = block_id
        self.layer = layer

        self.start = np.array(start)
        self.end = np.array(end)
        self.center = (self.start + self.end) / 2
        
        self.active = False
        self.nodes = []
        self.temperature = None

    def __repr__(self):

        return f"Block(id={self.id}, layer={self.layer})"


class Geometry:

    def __init__(self, stl_file):

        self.mesh = trimesh.load_mesh(stl_file)

        self.bounds = self.mesh.bounds

        self.blocks = []

    # --------------------------------------------------------------

    def build_blocks(self):

        """
        Divide the part:
        layer -> hatch -> 5 blocks

        paper Section 4.3.1.1
        """

        xmin, ymin, zmin = self.bounds[0]
        xmax, ymax, zmax = self.bounds[1]

        block_length = BLOCK["length"]
        layer_height = BLOCK["height"]

        total_layers = BUILD["layers"]

        block_id = 0

        z = zmin

        for layer in range(total_layers):

            # alternate scan direction

            if layer % 2 == 0:

                x_positions = np.arange(
                    xmin,
                    xmax,
                    block_length
                )

            else:

                x_positions = np.arange(
                    xmax - block_length,
                    xmin - block_length,
                    -block_length
                )

            for x in x_positions:

                start = (
                    x,
                    ymin,
                    z
                )

                end = (
                    x + block_length,
                    ymax,
                    z + layer_height
                )

                self.blocks.append(

                    Block(
                        block_id,
                        layer,
                        start,
                        end
                    )

                )

                block_id += 1

            z += layer_height

        print("------------------------------------")
        print("Geometry created")
        print("------------------------------------")
        print("Layers :", total_layers)
        print("Blocks :", len(self.blocks))
        print()

    # --------------------------------------------------------------

    def get_block(self, idx):

        return self.blocks[idx]

    # --------------------------------------------------------------

    def activate(self, idx):

        self.blocks[idx].active = True

    # --------------------------------------------------------------

    def deactivate_all(self):

        for block in self.blocks:

            block.active = False

    # --------------------------------------------------------------

    def active_blocks(self):

        return [b for b in self.blocks if b.active]

    # --------------------------------------------------------------

    def deposition_order(self):

        """
        Returns blocks exactly
        in print order.
        """

        return self.blocks


# ------------------------------------------------------------------

if __name__ == "__main__":

  
    stl_file = BUILD["stl_file"]
    
    geom = Geometry(stl_file)

    geom.build_blocks()

    print(geom.blocks[0])

    print(geom.blocks[-1])