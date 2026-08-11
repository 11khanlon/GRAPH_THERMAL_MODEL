import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os 
import time

# ----------------------------------
def draw_block(ax, block, color, alpha, edge=True, edge_color=None):

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

    ax.add_collection3d(cube)

#-------------------------------------
def draw_mask(
    ax,
    xmin,
    xmax,
    ymin,
    ymax,
    z,
    color="royalblue",
    alpha=1.0
):
    """
    Draw an opaque rectangular mask at height z.

    Visualization only. Does not modify geometry.
    """

    vertices = np.array([
        [xmin, ymin, z],
        [xmax, ymin, z],
        [xmax, ymax, z],
        [xmin, ymax, z],
    ])

    face = Poly3DCollection(
        [vertices],
        facecolors=color,
        edgecolors="none",
        alpha=alpha
    )

    ax.add_collection3d(face)

#------------------------------

def draw_box(
    ax,
    start,
    end,
    color,
    alpha=1.0,
    edge_color="black"
):

    """
    Draw one rectangular box.

    Used only for visualization.
    """

    x0, y0, z0 = start
    x1, y1, z1 = end

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
        edgecolors=edge_color,
        linewidths=0.8,
        alpha=alpha,
    )

    ax.add_collection3d(cube)


def plot_geometry_clean(geometry):

    """
    Clean visualization of the geometry.

    Visualization only:
        - Substrate is shown as ONE solid rectangular block.
        - Deposition is shown using the discretized blocks.
        - STL can optionally be shown.
    
    This does NOT modify the geometry, nodes, or thermal model.
    """

    fig = plt.figure(figsize=(12, 9))

    ax = fig.add_subplot(111, projection="3d")



    deposition_blocks = [
    block for block in geometry.blocks
    if not block.is_substrate]

    first_layer_number = min(
        block.layer
        for block in deposition_blocks
    )

    first_layer_blocks = [
        block for block in deposition_blocks
        if block.layer == first_layer_number
    ]


    first_xmin = min(
    block.start[0]
    for block in first_layer_blocks
    )

    first_xmax = max(
        block.end[0]
        for block in first_layer_blocks
    )

    first_ymin = min(
        block.start[1]
        for block in first_layer_blocks
    )

    first_ymax = max(
        block.end[1]
        for block in first_layer_blocks
    )


    substrate_blocks = [
    block for block in geometry.blocks
    if block.is_substrate
    ]

    substrate_top = max(
        block.end[2]
        for block in substrate_blocks
    )

    draw_mask(
        ax,
        first_xmin,
        first_xmax,
        first_ymin,
        first_ymax,
        substrate_top + 1e-9,
        color="royalblue",
        alpha=1.0
    )

    # ==========================================================
    # 1. SUBSTRATE BOUNDS
    # ==========================================================

    substrate_blocks = [
        block for block in geometry.blocks
        if block.is_substrate
    ]

    substrate_xmin = min(
        block.start[0] for block in substrate_blocks
    )

    substrate_xmax = max(
        block.end[0] for block in substrate_blocks
    )

    substrate_ymin = min(
        block.start[1] for block in substrate_blocks
    )

    substrate_ymax = max(
        block.end[1] for block in substrate_blocks
    )

    substrate_zmin = min(
        block.start[2] for block in substrate_blocks
    )

    substrate_zmax = max(
        block.end[2] for block in substrate_blocks
    )

    # ==========================================================
    # 2. DRAW SUBSTRATE AS ONE BLOCK
    # ==========================================================

    draw_box(
        ax,
        (
            substrate_xmin,
            substrate_ymin,
            substrate_zmin
        ),
        (
            substrate_xmax,
            substrate_ymax,
            substrate_zmax
        ),
        color="royalblue",
        alpha=0.25,
        edge_color="black"
    )

    # ==========================================================
    # 3. DRAW ONLY DEPOSITION BLOCKS
    # ==========================================================

    for block in geometry.blocks:

        if not block.is_substrate:

            draw_block(
                ax,
                block,
                "crimson",
                alpha=1.0,
                edge=True,
                edge_color="black"
            )

    # ==========================================================
    # 4. PLOTTING LIMITS
    # ==========================================================

    ax.set_xlim(
        substrate_xmin,
        substrate_xmax
    )

    ax.set_ylim(
        substrate_ymin,
        substrate_ymax
    )

    ax.set_zlim(
        substrate_zmin,
        max(
            block.end[2]
            for block in geometry.blocks
        )
    )

    # ==========================================================
    # 5. LABELS
    # ==========================================================

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title(
        "DED Geometry — Discretized Deposition on Substrate"
    )

    # ==========================================================
    # 6. ASPECT RATIO
    # ==========================================================

    ax.set_box_aspect(
        (
            substrate_xmax - substrate_xmin,
            substrate_ymax - substrate_ymin,
            substrate_zmax - substrate_zmin
            + geometry.height
        )
    )

    plt.tight_layout()

    plt.show()