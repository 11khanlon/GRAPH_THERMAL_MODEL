#Thermocouple and numerical probe definitions.

import numpy as np


# ---------------------------------------------------------
# Thermal probe class
class ThermalProbe:

    def __init__(self,
                 name,
                 position,
                 region="substrate", 
                 layer = None):

        self.name = name
        self.position = np.asarray(position, dtype=float)
        self.region = region
        self.layer = layer 

        self.node = None
        self.node_id = None

        self.original_position = None
        self.snap_distance = None


    # -----------------------------------------------------

    def assign_node(self, nodes, snap_to_position=True):

        candidates = self._candidate_nodes(nodes)

        if len(candidates) == 0:

            raise ValueError(f"No candidate nodes available for probe {self.name}")

        distances = np.array([
            np.linalg.norm(node.position - self.position) 
            for node in candidates])

        nearest_index = np.argmin(distances)
        node = candidates[nearest_index]

        self.node = node
        self.node_id = node.id

        self.original_position = (node.position.copy())
        self.snap_distance = (distances[nearest_index])

        if snap_to_position:

            node.position = (self.position.copy())

        return node


     #----------- TC0, TC1, TC2, TC3 -------------------
    
    def temperature_from_nearest_node(
        self,
        nodes,
        temperature,
        active_only=False):

        candidates = self._candidate_nodes(nodes)

        if active_only:

            candidates = [
                node
                for node in candidates
                if node.active
            ]

        if len(candidates) == 0:

            return np.nan

        distances = np.array([
            np.linalg.norm(node.position - self.position)
            for node in candidates])

        nearest = candidates[np.argmin(distances)]

        return temperature[nearest.id]

    def interpolate_temperature(
        self,
        nodes,
        temperature,
        k=8,
        power=2,
        active_only=False):

        candidates = self._candidate_nodes(nodes)

        if active_only:

            candidates = [
                node
                for node in candidates
                if node.active
            ]

        if len(candidates) == 0:

            return np.nan

        positions = np.array([
            node.position
            for node in candidates
        ])

        node_ids = np.array([
            node.id
            for node in candidates
        ])

        distances = np.linalg.norm(positions - self.position, axis = 1)
        k_use = min(k, len(candidates))
        nearest = np.argsort(distances)[:k_use]
        d = distances[nearest]
        ids = node_ids[nearest]

        # Exact node coincidence
        if np.any(d < 1e-12):

            return temperature[ids[np.argmin(d)]]

        weights = (1.0/ d**power)

        return (np.sum(weights * temperature[ids]) / np.sum(weights))
    

    def _candidate_nodes(self, nodes):

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

            if self.layer is not None:

                candidates = [
                    node
                    for node in candidates
                    if node.block.layer
                    == self.layer
                    ]

        elif self.region == "all":

            candidates = list(nodes)

        else:

            raise ValueError(
                f"Unknown probe region: "
                f"{self.region}"
            )

        return candidates

    # -----------------------------------------------------
    def summary(self):

        print(f"\n------ PROBE: {self.name} ------")
        print("Target position [mm]:", self.position * 1000)
        print("Region:", self.region)

        if self.layer is not None:
            print("Layer restriction:", self.layer)

        # Snapped-node mode
        if self.node is not None:

            print("Node ID:", self.node_id)
            print("Node block:", self.node.block_id)
            print("Node layer:", self.node.block.layer)
            print("Substrate:", self.node.is_substrate)
            print("Active:", self.node.active)

            if self.original_position is not None:
                print("Original node position [mm]:", self.original_position * 1000)

            if self.snap_distance is not None:
                print("Snap distance [mm]:", self.snap_distance * 1000)

            print("Final node position [mm]:", self.node.position * 1000)

            print("Sampling mode: snapped graph node")

        
        # Passive interpolation mode
        else:

            print("Node ID: None")
            print("Sampling mode: passive interpolation")
            print("Graph modified by probe: False")
        print("--------------------------------")

    # ============================================================
# SELECT MIDDLE BLOCK OF A DEPOSITION LAYER
# ============================================================

def middle_block_of_layer(geometry, layer=0): 

    #Return the middle deposition block from a selected layer 
   

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

def check_probe_location(probe, geometry):

    x, y, z = probe.position

    # Substrate bounds
    inside_substrate = (
        geometry.substrate_xmin <= x <= geometry.substrate_xmax
        and geometry.substrate_ymin <= y <= geometry.substrate_ymax
        and geometry.substrate_bottom <= z <= geometry.substrate_top
    )

    # Deposition footprint
    deposition_blocks = [
        block
        for block in geometry.blocks
        if not block.is_substrate
    ]

    dep_xmin = min(block.start[0] for block in deposition_blocks)
    dep_xmax = max(block.end[0] for block in deposition_blocks)
    dep_ymin = min(block.start[1] for block in deposition_blocks)
    dep_ymax = max(block.end[1] for block in deposition_blocks)

    under_wall = (
        dep_xmin <= x <= dep_xmax
        and dep_ymin <= y <= dep_ymax
    )

    print(f"\n------ PROBE LOCATION: {probe.name} ------")
    print("Position [mm]:", probe.position * 1000)
    print("Region:", probe.region)
    print("Inside substrate:", inside_substrate)
    print("Within wall XY footprint:", under_wall)
    print(
        "Depth below substrate top [mm]:",
        (geometry.substrate_top - z) * 1000
    )
    print("----------------------------------")

    return inside_substrate if probe.region == "substrate" else under_wall
   

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
    from nodes import NodeGenerator
    from material import Al5183Material, Al6061Material, AluminumMaterial

    from config import BUILD, SENSORS, SENSORS

    stl_file = BUILD["stl_file"]


    substrate_material = Al6061Material()
    wire_material = Al5183Material()

    geometry = Geometry(stl_file, substrate_material=substrate_material, wire_material=wire_material)
    geometry.build_blocks()
    generator = NodeGenerator(geometry)
    generator.generate()
    nodes = generator.nodes

    #------------------------------------
    #Thermocouple A test 

    test_position = np.array([
        0,
        50e-3,
        geometry.substrate_top - 0.1e-3
    ])

    probe = ThermalProbe(
        name="Test_TC",
        position=test_position,
        region="substrate"
    )

    probe.summary()

    check_probe_location(
        probe,
        geometry
    )

    plot_probe_location(
        geometry=geometry,
        nodes=nodes,
        probe=probe,
        show_nodes=True
    )

    #---------------------------------------
    #Meltpool test location 

    middle_block = middle_block_of_layer(
    geometry,
    layer=0
    )

    meltpool_probe = ThermalProbe(
        name="MeltPoolProbe",
        position=middle_block.center,
        region="deposition",
        layer=0
    )

    meltpool_probe.summary()

    check_probe_location(
        meltpool_probe,
        geometry
    )

    plot_probe_location(
        geometry=geometry,
        nodes=nodes,
        probe=meltpool_probe,
        show_nodes=True
    )

    # Initialize test temperatures
    T_test = np.full(
        len(nodes),
        25.0
    )

    # Test interpolation
    T_TC = probe.interpolate_temperature(
        nodes=nodes,
        temperature=T_test,
        k=8,
        active_only=True
    )

    print("Thermocouple temperature:", T_TC)


    print(
        "Total substrate nodes:",
        sum(node.is_substrate for node in nodes)
    )

    print(
        "Active substrate nodes:",
        sum(
            node.is_substrate and node.active
            for node in nodes
        )
    )

