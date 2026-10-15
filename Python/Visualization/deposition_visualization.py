"""Animate saved WAAM/DED thermal history with 3D and top-down views.

Run from Python/Visualization/ after main.py saves thermal_history.npz.
No direct dependency on DepositionTracker is needed.
"""
from pathlib import Path
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle


VIS_DIR = Path(__file__).resolve().parent
PYTHON_DIR = VIS_DIR.parent
GRAPH_CODE_DIR = PYTHON_DIR / "GraphCode"
for directory in (PYTHON_DIR, GRAPH_CODE_DIR):
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from GraphCode.material import WireMaterial, SubstrateMaterial

# Optional: drawing can use geometry blocks if the model is importable.
try:
    from GraphCode.geometry import Geometry
    from GraphCode.config import BUILD
except ImportError as exc:
    Geometry = None
    BUILD = None
    warnings.warn(f"Geometry unavailable; plotting from saved nodes only: {exc}")


def _read(data, names, required=False):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    if required:
        raise KeyError(f"Missing saved array {names}. Available keys: {data.files}")
    return None


def load_history(path):
    """Read arrays saved by main.py; never regenerate random nodes."""
    with np.load(path, allow_pickle=False) as data:
        print("Saved NPZ keys:", data.files)
        T = _read(data, ("temperature", "temperature_history", "history"), True).astype(float)
        times = _read(data, ("time", "time_history", "history_time"), True).astype(float)
        xyz = _read(data, ("node_positions", "positions"), True).astype(float)
        active = _read(data, ("active_history", "activation_history"))
        substrate = _read(data, ("substrate_mask", "is_substrate"))
        block_ids = _read(data, ("node_block_ids", "block_ids"))
        source = _read(data, ("laser_position_history", "heat_source_position_history", "source_position_history", "arc_position_history"))

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("node_positions must be (N, 3)")
    if T.shape != (len(times), len(xyz)):
        raise ValueError(f"Temperature shape {T.shape} must be ({len(times)}, {len(xyz)})")
    if active is None:
        warnings.warn("No active_history saved: all nodes will be shown, including potentially future deposition nodes.")
        active = np.ones_like(T, dtype=bool)
    active = np.asarray(active, dtype=bool)
    if active.shape != T.shape:
        raise ValueError("active_history must have the same shape as temperature")
    if substrate is not None:
        substrate = np.asarray(substrate, dtype=bool).reshape(-1)
        if len(substrate) != len(xyz):
            raise ValueError("substrate_mask length must match node count")
    if block_ids is not None and len(block_ids) != len(xyz):
        raise ValueError("node_block_ids length must match node count")
    if source is None:
        source = np.full((len(times), 3), np.nan)
        warnings.warn("No heat-source positions saved: arc marker will be hidden")
    source = np.asarray(source, dtype=float)
    if source.shape != (len(times), 3):
        raise ValueError("heat-source position history must be (number_of_frames, 3)")
    return times, T, xyz, active, substrate, block_ids, source


def _draw_block_edges(ax, block):
    lo = np.asarray(block.start, dtype=float)
    hi = np.asarray(block.end, dtype=float)
    corners = np.array([[lo[0],lo[1],lo[2]], [hi[0],lo[1],lo[2]],
                        [hi[0],hi[1],lo[2]], [lo[0],hi[1],lo[2]],
                        [lo[0],lo[1],hi[2]], [hi[0],lo[1],hi[2]],
                        [hi[0],hi[1],hi[2]], [lo[0],hi[1],hi[2]]])
    for i, j in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)):
        ax.plot(*corners[[i,j]].T, color="0.35", linewidth=.45, alpha=.55)


def animate_thermal_history(geometry, node_positions, temperature_history,
                            time_history, active_history, laser_position_history,
                            substrate_mask=None, node_block_ids=None,
                            ambient_temperature=25.0, maximum_temperature=None,
                            interval=150, deposition_only=False, frame_stride=1):
    """Return (figure, animation) with synchronized 3D and XY top-down panels."""
    xyz = np.asarray(node_positions, float)
    T = np.asarray(temperature_history, float)
    times = np.asarray(time_history, float)
    active = np.asarray(active_history, bool)
    source = np.asarray(laser_position_history, float)
    n_frames, n_nodes = T.shape
    if xyz.shape != (n_nodes, 3) or len(times) != n_frames or active.shape != T.shape or source.shape != (n_frames,3):
        raise ValueError("Inconsistent node/history/source array dimensions")
    if substrate_mask is None:
        if deposition_only:
            raise ValueError("deposition_only requires substrate_mask saved in NPZ")
        substrate = np.zeros(n_nodes, bool)
    else:
        substrate = np.asarray(substrate_mask, bool)
        if substrate.shape != (n_nodes,):
            raise ValueError("substrate_mask must be (N,)")
    block_ids = None if node_block_ids is None else np.asarray(node_block_ids)
    if block_ids is not None and block_ids.shape != (n_nodes,):
        raise ValueError("node_block_ids must be (N,)")
    if maximum_temperature is None:
        maximum_temperature = float(np.nanmax(T[active])) if np.any(active) else ambient_temperature + 1
    if maximum_temperature <= ambient_temperature:
        maximum_temperature = ambient_temperature + 1
    norm = Normalize(vmin=ambient_temperature, vmax=maximum_temperature)
    cmap = plt.get_cmap("turbo")
    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    ax3d = fig.add_subplot(121, projection="3d")
    axtop = fig.add_subplot(122)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax3d, axtop], shrink=.8, pad=.02, label="Temperature (°C)")
    limits = [(float(xyz[:, k].min()), float(xyz[:, k].max())) for k in range(3)]
    if geometry is not None:
        limits[0] = (geometry.substrate_xmin, geometry.substrate_xmax)
        limits[1] = (geometry.substrate_ymin, geometry.substrate_ymax)
        limits[2] = (geometry.substrate_bottom, geometry.zmax)
    ranges = [max(b-a, 1e-5) for a,b in limits]
    dep_blocks = [] if geometry is None else [b for b in geometry.blocks if not b.is_substrate]
    lookup = {int(b.id): b for b in dep_blocks} if dep_blocks else {}

    def update(frame):
        ax3d.cla()
        axtop.cla()
        visible = active[frame] & (~substrate if deposition_only else np.ones(n_nodes, bool))
        # Always show active substrate in the full 3D view; top view uses same nodes.
        if geometry is not None:
            axtop.add_patch(Rectangle((limits[0][0], limits[1][0]),
                         ranges[0], ranges[1], facecolor="0.93", edgecolor="0.4", zorder=0))
            if deposition_only:
                if block_ids is not None:
                    active_ids = np.unique(block_ids[visible])
                    for bid in active_ids:
                        block = lookup.get(int(bid))
                        if block is not None:
                            _draw_block_edges(ax3d, block)
                            x0,y0,_ = block.start
                            x1,y1,_ = block.end
                            axtop.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,
                                              fill=False,edgecolor="0.35",linewidth=.6))
                else:
                    for block in dep_blocks:
                        x0,y0,_ = block.start
                        x1,y1,_ = block.end
                        axtop.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,
                                          fill=False,edgecolor="0.75",linewidth=.5))
        if np.any(visible):
            ax3d.scatter(xyz[visible,0], xyz[visible,1], xyz[visible,2],
                         c=T[frame,visible], cmap=cmap, norm=norm,
                         s=25 if deposition_only else 9, depthshade=False)
            axtop.scatter(xyz[visible,0], xyz[visible,1],
                          c=T[frame,visible], cmap=cmap, norm=norm,
                          s=25 if deposition_only else 9, linewidths=0)
        if np.all(np.isfinite(source[frame])):
            x,y,z = source[frame]
            ax3d.scatter([x],[y],[z],color="red",marker="*",s=120,depthshade=False)
            axtop.scatter([x],[y],color="red",marker="*",s=125,edgecolors="black",linewidths=.4,zorder=5)
        ax3d.set(xlim=limits[0],ylim=limits[1],zlim=limits[2],
                 xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",
                 title=f"3D thermal field | t = {times[frame]:.2f} s")
        ax3d.set_box_aspect(ranges)
        ax3d.view_init(elev=23, azim=-65)
        axtop.set(xlim=limits[0],ylim=limits[1],xlabel="X (m)",ylabel="Y (m)",
                  title="Top-down view (XY)")
        axtop.set_aspect("equal", adjustable="box")
        axtop.grid(alpha=.15)
        return ()

    animation = FuncAnimation(fig, update, frames=range(0,n_frames,max(1,frame_stride)),
                              interval=interval, repeat=False)
    return fig, animation


def animate_deposition_view(geometry, node_positions, temperature_history,
                           time_history, active_history, laser_position_history,
                           substrate_mask, node_block_ids, **kwargs):
    return animate_thermal_history(
        geometry, node_positions, temperature_history, time_history,
        active_history, laser_position_history, substrate_mask, node_block_ids,
        deposition_only=True, **kwargs)


if __name__ == "__main__":
    # Adjust this location if main.py saves to a different directory.
    history_file = PYTHON_DIR / "thermal_history.npz"
    if not history_file.exists():
        raise FileNotFoundError(f"History not found: {history_file}. Update history_file to its actual location.")
    times, T, xyz, active, substrate, block_ids, source = load_history(history_file)
    geometry = None
    if Geometry is not None and BUILD is not None:
        try:
            geometry = Geometry(
                    BUILD["stl_file"],
                    substrate_material=SubstrateMaterial(),
                    wire_material=WireMaterial()
                )
            geometry.build_blocks()
        except Exception as exc:
            warnings.warn(f"Could not recreate geometry; using node bounds: {exc}")
    # Set True to show only deposition nodes; requires saved substrate_mask.
    DEPOSITION_ONLY = substrate is not None
    fig, animation = animate_thermal_history(
        geometry, xyz, T, times, active, source,
        substrate_mask=substrate, node_block_ids=block_ids,
        deposition_only=DEPOSITION_ONLY, frame_stride=1)
    gif_file = PYTHON_DIR / "ded_thermal_animation.gif"
    animation.save(gif_file, writer=PillowWriter(fps=5), dpi=100)
    print(f"Saved animation: {gif_file}")
    plt.show()
