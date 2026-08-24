"""
Thermocouple and numerical probe definitions.

Sensors do not create additional graph nodes.

Instead, each sensor is associated with an existing
thermal node. That node may optionally be moved to the
exact sensor coordinate before graph construction.
"""

import numpy as np


# ---------------------------------------------------------
# THERMAL PROBE
# ---------------------------------------------------------

class ThermalProbe:

    def __init__(
        self,
        name,
        position,
        region="substrate"
    ):

        self.name = name

        self.position = np.asarray(
            position,
            dtype=float
        )

        self.region = region

        self.node = None
        self.node_id = None

        self.original_position = None
        self.snap_distance = None


    # -----------------------------------------------------

    def assign_node(
        self,
        nodes,
        snap_to_position=True
    ):

        """
        Find the existing thermal node nearest the
        requested probe position.

        region options:
            "substrate"
            "deposition"
            "all"
        """

        if self.region == "substrate":

            candidates = [
                node
                for node in nodes
                if node.is_substrate
            ]

        elif self.region == "deposition":

            candidates = [
                node
                for node in nodes
                if not node.is_substrate
            ]

        elif self.region == "all":

            candidates = nodes

        else:

            raise ValueError(
                f"Unknown probe region: {self.region}"
            )

        if len(candidates) == 0:

            raise ValueError(
                f"No candidate nodes available "
                f"for probe {self.name}"
            )

        distances = np.array([
            np.linalg.norm(
                node.position
                - self.position
            )
            for node in candidates
        ])

        nearest_index = np.argmin(
            distances
        )

        node = candidates[
            nearest_index
        ]

        self.node = node
        self.node_id = node.id

        self.original_position = (
            node.position.copy()
        )

        self.snap_distance = (
            distances[nearest_index]
        )

        # Move existing graph node to exact
        # thermocouple/probe coordinate
        if snap_to_position:

            node.position = (
                self.position.copy()
            )

        return node

    


    # -----------------------------------------------------

    def summary(self):

        print(
            f"\n------ PROBE: {self.name} ------"
        )

        print(
            "Target position [mm]:",
            self.position * 1000
        )

        print(
            "Node ID:",
            self.node_id
        )

        if self.node is not None:

            print(
                "Node block:",
                self.node.block_id
            )

            print(
                "Node layer:",
                self.node.block.layer
            )

            print(
                "Substrate:",
                self.node.is_substrate
            )

            print(
                "Active:",
                self.node.active
            )

        if self.original_position is not None:

            print(
                "Original node position [mm]:",
                self.original_position * 1000
            )

        if self.snap_distance is not None:

            print(
                "Snap distance [mm]:",
                self.snap_distance * 1000
            )

        print(
            "Final node position [mm]:",
            self.node.position * 1000
        )

        print("--------------------------------")

    # ============================================================
# SELECT MIDDLE BLOCK OF A DEPOSITION LAYER
# ============================================================

def middle_block_of_layer(
    geometry,
    layer=0
):
    """
    Return the middle deposition block
    from a selected layer.
    """

    layer_blocks = [
        block
        for block in geometry.blocks
        if (
            not block.is_substrate
            and block.layer == layer
        )
    ]

    if len(layer_blocks) == 0:

        raise ValueError(
            f"No deposition blocks found "
            f"for layer {layer}."
        )

    # Sort from left to right
    layer_blocks.sort(
        key=lambda block:
            block.start[0]
    )

    middle_block = (
        layer_blocks[
            len(layer_blocks) // 2
        ]
    )

    return middle_block

def check_probe_location(
    probe,
    geometry
):

    x, y, z = probe.position

    deposition_blocks = [
        block
        for block in geometry.blocks
        if not block.is_substrate
    ]

    dep_xmin = min(
        block.start[0]
        for block in deposition_blocks
    )

    dep_xmax = max(
        block.end[0]
        for block in deposition_blocks
    )

    dep_ymin = min(
        block.start[1]
        for block in deposition_blocks
    )

    dep_ymax = max(
        block.end[1]
        for block in deposition_blocks
    )

    print(
        "\n------ PROBE LOCATION CHECK ------"
    )

    print(
        "Probe [mm]:",
        probe.position * 1000
    )

    print(
        "Deposition X [mm]:",
        dep_xmin * 1000,
        "to",
        dep_xmax * 1000
    )

    print(
        "Deposition Y [mm]:",
        dep_ymin * 1000,
        "to",
        dep_ymax * 1000
    )

    under_wall_x = (
        dep_xmin <= x <= dep_xmax
    )

    under_wall_y = (
        dep_ymin <= y <= dep_ymax
    )

    print(
        "Inside deposition X footprint:",
        under_wall_x
    )

    print(
        "Inside deposition Y footprint:",
        under_wall_y
    )

    print(
        "Depth below substrate top [mm]:",
        (geometry.substrate_top - z)
        * 1000
    )

    print("----------------------------------")

    return (
        under_wall_x
        and under_wall_y
    )


# ============================================================
# PLOT PROBE LOCATION
# ============================================================

def plot_probe_location(
    geometry,
    nodes,
    probe,
    show_nodes=True
):
    """
    Plot substrate/deposition nodes and highlight the
    selected thermocouple/probe location.

    Intended for validation/debugging only.
    """

    import matplotlib.pyplot as plt

    fig = plt.figure(
        figsize=(11, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # --------------------------------------------------------
    # Separate substrate and deposition nodes
    # --------------------------------------------------------

    substrate_nodes = np.array([
        node.position
        for node in nodes
        if node.is_substrate
    ])

    deposition_nodes = np.array([
        node.position
        for node in nodes
        if not node.is_substrate
    ])

    # --------------------------------------------------------
    # Plot graph nodes
    # --------------------------------------------------------

    if show_nodes:

        if len(substrate_nodes) > 0:

            ax.scatter(
                substrate_nodes[:, 0],
                substrate_nodes[:, 1],
                substrate_nodes[:, 2],
                s=4,
                alpha=0.20,
                label="Substrate nodes"
            )

        if len(deposition_nodes) > 0:

            ax.scatter(
                deposition_nodes[:, 0],
                deposition_nodes[:, 1],
                deposition_nodes[:, 2],
                s=10,
                alpha=0.50,
                label="Deposition nodes"
            )

    # --------------------------------------------------------
    # Plot thermocouple / probe
    # --------------------------------------------------------

    x, y, z = probe.position

    ax.scatter(
        [x],
        [y],
        [z],
        s=150,
        marker="*",
        label=probe.name
    )

    # --------------------------------------------------------
    # Add text label beside probe
    # --------------------------------------------------------

    ax.text(
        x,
        y,
        z,
        (
            f"  {probe.name}\n"
            f"  ({x*1000:.2f}, "
            f"{y*1000:.2f}, "
            f"{z*1000:.2f}) mm"
        ),
        fontsize=10
    )

    # --------------------------------------------------------
    # Draw deposition footprint on substrate top
    # --------------------------------------------------------

    deposition_blocks = [
        block
        for block in geometry.blocks
        if not block.is_substrate
    ]

    if deposition_blocks:

        dep_xmin = min(
            block.start[0]
            for block in deposition_blocks
        )

        dep_xmax = max(
            block.end[0]
            for block in deposition_blocks
        )

        dep_ymin = min(
            block.start[1]
            for block in deposition_blocks
        )

        dep_ymax = max(
            block.end[1]
            for block in deposition_blocks
        )

        z_top = geometry.substrate_top

        # Rectangle showing wall footprint
        x_outline = [
            dep_xmin,
            dep_xmax,
            dep_xmax,
            dep_xmin,
            dep_xmin
        ]

        y_outline = [
            dep_ymin,
            dep_ymin,
            dep_ymax,
            dep_ymax,
            dep_ymin
        ]

        z_outline = [
            z_top
            for _ in x_outline
        ]

        ax.plot(
            x_outline,
            y_outline,
            z_outline,
            linewidth=2,
            label="Deposition footprint"
        )

    # --------------------------------------------------------
    # Axis bounds
    # --------------------------------------------------------

    ax.set_xlim(
        geometry.substrate_xmin,
        geometry.substrate_xmax
    )

    ax.set_ylim(
        geometry.substrate_ymin,
        geometry.substrate_ymax
    )

    ax.set_zlim(
        geometry.substrate_bottom,
        geometry.zmax
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title(
        f"Thermal Probe Location: {probe.name}"
    )

    ax.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from Validation.visualization import  plot_geometry_with_nodes
    from face import update_exposed_faces
    from face import update_exposed_faces, print_exposed_faces, print_exposure_statistics
    from config import BLOCK,BUILD

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


    surface_thickness = BLOCK["height"]/2
    generator.update_node_exposure(surface_thickness)

    print(generator.nodes[0])

    generator.validate_nodes()

    #plot_geometry_with_nodes(geom, generator)

    print_surface_statistics(nodes)