
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os 
import time

# ------------------------------
def draw_block(ax, block, color):

    """
    Draw one block as a transparent rectangular prism.
    """

    x0, y0, z0 = block.start
    x1, y1, z1 = block.end

    vertices = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])

    faces = [

        [vertices[0], vertices[1], vertices[2], vertices[3]],

        [vertices[4], vertices[5], vertices[6], vertices[7]],

        [vertices[0], vertices[1], vertices[5], vertices[4]],

        [vertices[2], vertices[3], vertices[7], vertices[6]],

        [vertices[1], vertices[2], vertices[6], vertices[5]],

        [vertices[0], vertices[3], vertices[7], vertices[4]],

    ]

    cube = Poly3DCollection(

        faces,

        facecolors=color,

        edgecolors="black",

        linewidths=0.25,

        alpha=0.20,

    )

    ax.add_collection3d(cube)

def plot_geometry(geometry):

    """
    Plot the STL together with every discretized block.
    """

    fig = plt.figure(figsize=(12,9))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

   
    # STL

    mesh = Poly3DCollection(

        geometry.mesh.triangles,

        facecolor="lightgray",

        edgecolor="black",

        alpha=0.10,

    )

    ax.add_collection3d(mesh)

  
    # Draw every block

    for block in geometry.blocks:

        if block.is_substrate:

            draw_block(
                ax,
                block,
                "royalblue"
            )

        else:

            draw_block(
                ax,
                block,
                "crimson"
            )

    

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title("Block Discretization")

    # Substrate bounds

    xmin = geometry.xmin - (geometry.substrate_length - geometry.length)/2
    xmax = xmin + geometry.substrate_length

    ymin = geometry.ymin - (geometry.substrate_width -  geometry.width)/2
    ymax = ymin + geometry.substrate_width

    ax.set_box_aspect(
    (xmax - xmin, ymax - ymin, geometry.zmax - (geometry.zmin - geometry.substrate_height)
    )
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(
        geometry.zmin - geometry.substrate_height,
        geometry.zmax)

    plt.tight_layout()

    plt.show()


def plot_nodes(
    self,
    show_substrate=True,
    show_deposition=True,
):
    """
    Display all generated nodes.

    Blue = substrate

    Red = deposition
    """

    fig = plt.figure(figsize=(10,8))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    substrate_x = []
    substrate_y = []
    substrate_z = []

    deposition_x = []
    deposition_y = []
    deposition_z = []

    for node in self.nodes:

        if node.is_substrate:

            substrate_x.append(node.position[0])
            substrate_y.append(node.position[1])
            substrate_z.append(node.position[2])

        else:

            deposition_x.append(node.position[0])
            deposition_y.append(node.position[1])
            deposition_z.append(node.position[2])

    if show_substrate:

        ax.scatter(
            substrate_x,
            substrate_y,
            substrate_z,
            s=2,
            c="blue",
            label="Substrate"
        )

    if show_deposition:

        ax.scatter(
            deposition_x,
            deposition_y,
            deposition_z,
            s=4,
            c="red",
            label="Deposition"
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.legend()

    plt.tight_layout()

    plt.show()