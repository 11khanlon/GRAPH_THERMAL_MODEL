import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os 
import time


# ----------------------------------
def draw_block(ax, block, color, alpha, edge=True, edge_color=None, zorder=0):

    #Draw one block as a transparent rectangular prism.
    
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
        [x0, y1, z1]
    ])

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]

    if edge:

        if edge_color is None:
            edge_color = "black"

        line_width = 0.25

    else:

        edge_color = "none"
        line_width = 0

    cube = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=edge_color,
        linewidths=line_width,
        alpha=alpha
    )

    cube.set_zorder(zorder) 
    ax.add_collection3d(cube)


#%%
def plot_nodes(
    node_generator,
    show_substrate=True,
    show_deposition=True,
):
    """
    Display generated graph nodes.

    Blue = substrate nodes
    Red  = deposition nodes
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


    for node in node_generator.nodes:


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
            s=3,
            c="blue",
            label="Substrate"
        )


    if show_deposition:

        ax.scatter(
            deposition_x,
            deposition_y,
            deposition_z,
            s=5,
            c="red",
            label="Deposition"
        )


    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


    ax.legend()


    plt.tight_layout()

    plt.show()



def plot_geometry_with_nodes(
        geometry,
        node_generator):

    fig = plt.figure(figsize=(12,9))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    # -----------------------------
    # Draw blocks
    # -----------------------------

    for block in geometry.blocks:


        if block.is_substrate:

            draw_block(
                ax,
                block,
                "grey",
                alpha=0.05,
                edge= False
            )


        else:

            draw_block(
                ax,
                block,
                "lightblue",
                alpha=0.15,
                edge= False
            )


    # -----------------------------
    # Draw nodes
    # -----------------------------


    substrate_nodes = []
    deposition_nodes = []


    # deposition bounds

    dep_xmin = geometry.xmin
    dep_xmax = geometry.xmax

    dep_ymin = geometry.ymin
    dep_ymax = geometry.ymax

    dep_zmin = geometry.zmin


    for node in node_generator.nodes:


        x,y,z = node.position


        if node.is_substrate:


            # Hide substrate nodes directly under deposited part

            underneath_deposition = (

            dep_xmin <= x <= dep_xmax

            and

            dep_ymin <= y <= dep_ymax

            )


            if not underneath_deposition:

                substrate_nodes.append(node.position)


        else:

            deposition_nodes.append(node.position)



    substrate_nodes = np.array(substrate_nodes)
    deposition_nodes = np.array(deposition_nodes)


    
    ax.scatter(
    substrate_nodes[:,0],
    substrate_nodes[:,1],
    substrate_nodes[:,2],
    s=1,
    c="black",
    alpha=0.25,
    label="Substrate Nodes"
    )


    ax.scatter(
        deposition_nodes[:,0],
        deposition_nodes[:,1],
        deposition_nodes[:,2],
        s=0.7,
        c="black",
        alpha=1.0,
        label="Deposition Nodes"
    )


    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


    ax.legend()


    # same physical scaling

    ax.set_box_aspect(
        (
            geometry.substrate_length,
            geometry.substrate_width,
            geometry.substrate_height + geometry.height
        )
    )


    plt.show()

