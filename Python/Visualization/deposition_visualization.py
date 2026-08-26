"""
Visualization tools for the transient DED thermal simulation.

Contains:
    - laser visualization
    - thermal-history animation
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry_visualization import draw_block


#---------------------
def draw_block_outline(
    ax,
    block,
    linewidth=1.0
):

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

    edges = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),

        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),

        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7)
    ]

    for i, j in edges:

        ax.plot(
            [vertices[i, 0], vertices[j, 0]],
            [vertices[i, 1], vertices[j, 1]],
            [vertices[i, 2], vertices[j, 2]],
            color="black",
            linewidth=linewidth
        )

# -----------------------------
# Draw a laser object  
def draw_laser(
    ax,
    position,
    radius=0.4e-3,
    height=8e-3):
    
    position = np.asarray(position, dtype=float)

    # No laser during dwell / inactive periods
    if np.any(np.isnan(position)):
        return

    x0, y0, z0 = position

    theta = np.linspace(0, 2 * np.pi, 30)
    z = np.linspace(z0, z0 + height,20)
    theta_grid, z_grid = np.meshgrid(theta, z)


    X = (x0 + radius * np.cos(theta_grid))
    Y = (y0 + radius * np.sin(theta_grid))
    Z = z_grid

    ax.plot_surface(X, Y, Z, color="red", alpha=0.65, linewidth = 0 )

#--------------------------
#Draw substrate surface 

def draw_substrate_surface(
    ax,
    geometry,
    alpha=1.0,
    z_offset= .8
):

    xmin = geometry.substrate_xmin
    xmax = geometry.substrate_xmax

    ymin = geometry.substrate_ymin
    ymax = geometry.substrate_ymax

    z = (
        geometry.substrate_top
        - z_offset
    )

    vertices = np.array([
        [xmin, ymin, z],
        [xmax, ymin, z],
        [xmax, ymax, z],
        [xmin, ymax, z]
    ])

    surface = Poly3DCollection(
        [vertices],
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
        alpha=alpha
    )

    surface.set_zorder(-10)

    ax.add_collection3d(
        surface
    )


#--------------------------
#Thermal history animation 

def animate_thermal_history(
    geometry,
    node_positions,
    temperature_history,
    time_history,
    active_history,
    laser_position_history,
    ambient_temperature = 25.0,
    maximum_temperature = 2200.0,
    interval = 150):


    # Convert input histories to arrays
    temperature_history = np.asarray(temperature_history)
    time_history = np.asarray(time_history)
    active_history = np.asarray(active_history,dtype=bool)
    laser_position_history = np.asarray(laser_position_history, dtype=float)


    substrate_mask = np.asarray(
        substrate_mask,
        dtype=bool
    )

    node_block_ids = np.asarray(
        node_block_ids,
        dtype=int
    )

    # Validation
    n_frames = len(time_history)

    if temperature_history.shape[0] != n_frames:

        raise ValueError(
            "temperature_history and time_history "
            "must contain the same number of states.")


    if active_history.shape[0] != n_frames:

        raise ValueError(
            "active_history and time_history "
            "must contain the same number of states.")


    if laser_position_history.shape[0] != n_frames:

        raise ValueError(
            "laser_position_history and time_history "
            "must contain the same number of states.")

    if len(substrate_mask) != n_nodes:

        raise ValueError(
            "substrate_mask length does not match "
            "the number of nodes."
        )


    if len(node_block_ids) != n_nodes:

        raise ValueError(
            "node_block_ids length does not match "
            "the number of nodes."
        )


    # Saved node positions
    node_positions = np.asarray(node_positions, dtype=float)

    if node_positions.ndim != 2 or node_positions.shape[1] != 3:

        raise ValueError(
            "node_positions must have shape (n_nodes, 3).")

    n_nodes = node_positions.shape[0]

    if temperature_history.shape[1] != n_nodes:

        raise ValueError(
            "temperature_history node dimension does "
            "not match node_positions.")


    if active_history.shape[1] != n_nodes:

        raise ValueError(
            "active_history node dimension does "
            "not match node_positions.")


    # Node coordinates
    positions = node_positions

    # Plot the figure 
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Temperature scale
    norm = Normalize(vmin=ambient_temperature, vmax=maximum_temperature)
    cmap = plt.get_cmap("turbo")

    # Color bar
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax, pad=0.10, shrink=0.75)
    colorbar.set_label("Temperature (°C)")


    # FRAME UPDATE
    def update(frame):

        ax.clear()

        # Current simulation state
        active_mask = (active_history[frame])
        T = (temperature_history[frame])

        # Substrate geometry
        for block in geometry.blocks:

            if block.is_substrate:

                draw_block(
                    ax,
                    block,
                    color="linen",
                    alpha=0.15,
                    edge=False)
                
        # Thermal graph nodes
        ax.scatter(positions[active_mask, 0],
            positions[active_mask, 1],
            positions[active_mask, 2],
            c = T[active_mask],
            cmap=cmap,
            norm=norm,
            s=8
            )
        

        # Laser
        laser_position = (laser_position_history[frame])
        draw_laser(ax, laser_position)


        # Labels
        ax.set_title("DED Thermal Simulation\n" f"t = {time_history[frame]:.3f} s")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Geometry limits
        ax.set_xlim(geometry.substrate_xmin, geometry.substrate_xmax)
        ax.set_ylim(geometry.substrate_ymin, geometry.substrate_ymax)
        ax.set_zlim(geometry.substrate_bottom, geometry.zmax)

        ax.set_box_aspect(
            (geometry.substrate_length,
             geometry.substrate_width,
             geometry.substrate_height + geometry.height
            )
        )


        return ax,


    # Animation
    animation = FuncAnimation(fig, update, frames=n_frames, interval=interval, repeat=False)

    return (fig, animation)


#--------------------------
# Animation with no substrate 

def animate_deposition_view(
    geometry,
    node_positions,
    temperature_history,
    time_history,
    active_history,
    laser_position_history,
    substrate_mask,
    node_block_ids,
    ambient_temperature=25.0,
    maximum_temperature=2200.0,
    interval=150
):

    # Convert input histories to arrays
    temperature_history = np.asarray(temperature_history)
    time_history = np.asarray(time_history)
    active_history = np.asarray(active_history,dtype=bool)
    laser_position_history = np.asarray(laser_position_history, dtype=float)
    node_positions = np.asarray(node_positions, dtype=float)
    substrate_mask = np.asarray(substrate_mask, dtype=bool)
    node_block_ids = np.asarray(node_block_ids, dtype=int)


    # Validation
    n_frames = len(time_history)
    n_nodes = (node_positions.shape[0])

    if temperature_history.shape[0] != n_frames:

        raise ValueError(
            "temperature_history and time_history "
            "must contain the same number of states.")


    if active_history.shape[0] != n_frames:

        raise ValueError(
            "active_history and time_history "
            "must contain the same number of states.")


    if laser_position_history.shape[0] != n_frames:

        raise ValueError(
            "laser_position_history and time_history "
            "must contain the same number of states.")

    if temperature_history.shape != (
            n_frames,
            n_nodes
        ):

            raise ValueError(
                "temperature_history shape mismatch."
            )


    if active_history.shape != (
        n_frames,
        n_nodes
    ):

        raise ValueError(
            "active_history shape mismatch."
        )


    if len(substrate_mask) != n_nodes:

        raise ValueError(
            "substrate_mask shape mismatch."
        )


    if len(node_block_ids) != n_nodes:

        raise ValueError(
            "node_block_ids shape mismatch."
        )


    # Saved node positions
    node_positions = np.asarray(node_positions, dtype=float)

    if node_positions.ndim != 2 or node_positions.shape[1] != 3:

        raise ValueError(
            "node_positions must have shape (n_nodes, 3).")

    n_nodes = node_positions.shape[0]

    if temperature_history.shape[1] != n_nodes:

        raise ValueError(
            "temperature_history node dimension does "
            "not match node_positions.")


    if active_history.shape[1] != n_nodes:

        raise ValueError(
            "active_history node dimension does "
            "not match node_positions.")

    #deposition_blocks = 
    deposition_blocks = [
    block
    for block in geometry.blocks
    if not block.is_substrate]

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

    # Node coordinates

    deposition_mask = ~substrate_mask

    block_lookup = {
            block.id: block
            for block in geometry.blocks
        }

    # Plot the figure 
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Temperature scale
    norm = Normalize(vmin=ambient_temperature, vmax=maximum_temperature)
    cmap = plt.get_cmap("turbo")

    # Color bar
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax, pad=0.10, shrink=0.75)
    colorbar.set_label("Temperature (°C)")


    # FRAME UPDATE
    def update(frame):

        ax.clear()

        # Current simulation state
        active_mask = (active_history[frame])
        T = (temperature_history[frame])

        # Substrate geometry
        draw_substrate_surface(
            ax,
            geometry,
            alpha=1.0
        )

        # Thermal graph nodes

        visible_mask = (
            active_history[frame]
            & deposition_mask
        )

       

        active_block_ids = np.unique(
            node_block_ids[
                visible_mask
            ]
        )

        for block_id in active_block_ids:

            block = block_lookup[
                int(block_id)
            ]

            draw_block_outline(
                ax,
                block,
                linewidth=1.0
            )

        # Thermal graph nodes
        scatter = ax.scatter(
            node_positions[visible_mask, 0],
            node_positions[visible_mask, 1],
            node_positions[visible_mask, 2],

            c=T[visible_mask],

            cmap=cmap,
            norm=norm,

            s=40,

            depthshade=False
        )

        scatter.set_zorder(10)

    
        # Laser
        laser_position = (laser_position_history[frame])
        draw_laser(ax, laser_position)


        # Labels
        ax.set_title("DED Thermal Simulation\n" f"t = {time_history[frame]:.3f} s")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Geometry limits
        padding = 2.0e-3

        ax.set_xlim(geometry.substrate_xmin, geometry.substrate_xmax)
        ax.set_ylim(geometry.substrate_ymin, geometry.substrate_ymax)

       

        ax.set_zlim(
            geometry.substrate_top - 0.25e-3,
            geometry.zmax + 0.5e-3
        )

        ax.set_box_aspect(
            (
                geometry.substrate_length,
                geometry.substrate_width,
                geometry.substrate_height
                + geometry.height
            )
        )


        return ax,


    # Animation
    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        repeat=False
    )

    

    return (fig, animation)
    
    



# ============================================================
# TEST / LOAD SAVED SIMULATION
# ============================================================

if __name__ == "__main__":

    import sys
    from pathlib import Path

    # --------------------------------------------------------
    # Project paths
    # --------------------------------------------------------

    VISUALIZATION_DIR = (
        Path(__file__).resolve().parent
    )

    PYTHON_DIR = (
        VISUALIZATION_DIR.parent
    )

    if str(PYTHON_DIR) not in sys.path:

        sys.path.insert(
            0,
            str(PYTHON_DIR)
        )

    from GraphCode.geometry import Geometry
    from GraphCode.config import BUILD, LASER
    
    # --------------------------------------------------------
    # Locate thermal-history file
    # --------------------------------------------------------

    

    history_file = (
        PYTHON_DIR
        / "thermal_history.npz"
    )

    print(
        "\nLooking for thermal history:"
    )

    print(
        history_file
    )

    if not history_file.exists():

        raise FileNotFoundError(
            f"\nthermal_history.npz was not found at:\n"
            f"{history_file}\n"
        )


    # --------------------------------------------------------
    # Load saved data
    # --------------------------------------------------------

    data = np.load(
        history_file,
        allow_pickle=True
    )

    print(
        "\n------ SAVED NPZ CONTENTS ------"
    )

    print(
        data.files
    )


    # --------------------------------------------------------
    # Required histories
    # --------------------------------------------------------

    temperature_history = np.asarray(
        data["temperature"],
        dtype=float
    )

    time_history = np.asarray(
        data["time"],
        dtype=float
    )

    active_history = np.asarray(
        data["active_history"],
        dtype=bool
    )

    laser_position_history = np.asarray(
        data["laser_position_history"],
        dtype=float
    )

    node_positions = np.asarray(
        data["node_positions"],
        dtype=float
    )

    substrate_mask = np.asarray(
        data["substrate_mask"],
        dtype=bool
    )

    node_block_ids = np.asarray(
    data["node_block_ids"],
    dtype=int
    )


    print(
        "\nTemperature history shape:",
        temperature_history.shape
    )

    print(
        "Time history shape:",
        time_history.shape
    )

    print(
        "Active history shape:",
        active_history.shape
    )

    print(
        "Laser position history shape:",
        laser_position_history.shape
    )

    print(
        "Node positions shape:",
        node_positions.shape
    )


    # --------------------------------------------------------
    # Recreate geometry / nodes
    #
    # IMPORTANT:
    # See note below about randomized nodes.
    # --------------------------------------------------------

    geometry = Geometry(
        BUILD["stl_file"]
    )

    geometry.build_blocks()

   

    node_positions = np.asarray(
            data["node_positions"],
            dtype=float
        )


    # --------------------------------------------------------
    # Create animation
    # --------------------------------------------------------

    '''fig, animation = (
        animate_thermal_history(

            geometry=geometry,

            node_positions=node_positions,

            temperature_history=
                temperature_history,

            time_history=
                time_history,

            active_history=
                active_history,

            laser_position_history=
                laser_position_history,

            ambient_temperature=25.0,

            maximum_temperature=2200.0
        )
    )'''

    fig, animation = animate_deposition_view(

    geometry=geometry,

    node_positions=node_positions,

    temperature_history=
        temperature_history,

    time_history=
        time_history,

    active_history=
        active_history,

    laser_position_history=
        laser_position_history,

    substrate_mask=
        substrate_mask,

    node_block_ids=
        node_block_ids)


    OUTPUT_DIR = PYTHON_DIR
    gif_file = (
    OUTPUT_DIR
            / "ded_thermal_animation.gif"
        )

    animation.save(
        gif_file,
        writer="pillow",
        fps=5,
        dpi=150
    )

    # Keep animation referenced until window closes
    plt.show()

    