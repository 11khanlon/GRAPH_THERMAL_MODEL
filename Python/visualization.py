
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os 
import time

# ------------------------------
def draw_block(ax, block, color, alpha, edge=True, edge_color=None):

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
        alpha=alpha,
    )

    ax.add_collection3d(cube)
#-----------------------------------
def draw_outline(ax, xmin, xmax, ymin, ymax, zmin, zmax, color="black", linewidth=1.5, linestyle="-"):

    vertices = np.array([

        [xmin,ymin,zmin],
        [xmax,ymin,zmin],
        [xmax,ymax,zmin],
        [xmin,ymax,zmin],

        [xmin,ymin,zmax],
        [xmax,ymin,zmax],
        [xmax,ymax,zmax],
        [xmin,ymax,zmax],

    ])


    edges = [

        (0,1),(1,2),(2,3),(3,0),

        (4,5),(5,6),(6,7),(7,4),

        (0,4),(1,5),(2,6),(3,7)

    ]


    for e in edges:

        ax.plot3D(
            vertices[list(e),0],
            vertices[list(e),1],
            vertices[list(e),2],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=100
        )


#---------------------------
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

        edgecolor="none",

        alpha=0.10,

    )

    ax.add_collection3d(mesh)

  
    # Draw every block
    # Draw substrate first (background)

    for block in geometry.blocks:

        if block.is_substrate:


            # First few substrate layers under deposition
            if block.layer >= -9:


                draw_block(
                    ax,
                    block,
                    "royalblue",
                    alpha=0.15,
                    edge=True,
                    edge_color="none"
                )


            # Far substrate
            else:


                draw_block(
                    ax,
                    block,
                    "royalblue",
                    alpha=0.5,
                    edge=True,
                    edge_color="black"
                )


        # ----------------------------
        # Deposition
        # ----------------------------

        else:


            draw_block(
                ax,
                block,
                "crimson",
                alpha=0.9,
                edge=True,
                edge_color="black"
            )

    all_x = []
    all_y = []
    all_z = []

    for block in geometry.blocks:

        all_x.extend([
            block.start[0],
            block.end[0]
        ])

        all_y.extend([
            block.start[1],
            block.end[1]
        ])

        all_z.extend([
            block.start[2],
            block.end[2]
        ])


    ax.set_xlim(
        min(all_x),
        max(all_x)
    )

    ax.set_ylim(
        min(all_y),
        max(all_y)
    )

    ax.set_zlim(
        min(all_z),
        max(all_z)
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
        (
            geometry.substrate_length,
            geometry.substrate_width,
            geometry.substrate_height + geometry.height
        )
    )



    plt.tight_layout()

    plt.show()

def check_geometry_locations(geometry):

    print("\n-----------------------------")
    print("GEOMETRY LOCATION CHECK")
    print("-----------------------------")

    substrate = [
        block for block in geometry.blocks
        if block.is_substrate
    ]

    deposition = [
        block for block in geometry.blocks
        if not block.is_substrate
    ]

    print("\nSubstrate:")
    print("xmin:", min(b.start[0] for b in substrate))
    print("xmax:", max(b.end[0] for b in substrate))

    print("ymin:", min(b.start[1] for b in substrate))
    print("ymax:", max(b.end[1] for b in substrate))

    print("zmin:", min(b.start[2] for b in substrate))
    print("zmax:", max(b.end[2] for b in substrate))


    print("\nDeposition:")
    print("xmin:", min(b.start[0] for b in deposition))
    print("xmax:", max(b.end[0] for b in deposition))

    print("ymin:", min(b.start[1] for b in deposition))
    print("ymax:", max(b.end[1] for b in deposition))

    print("zmin:", min(b.start[2] for b in deposition))
    print("zmax:", max(b.end[2] for b in deposition))

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