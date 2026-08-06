import pyvista as pv


# ---------------------------------------------------
# Plot geometry using PyVista
# ---------------------------------------------------

def plot_geometry(geometry):

    """
    Plot substrate and deposition blocks.

    Substrate:
        Blue, transparent

    Deposition:
        Red, opaque
    """

    plotter = pv.Plotter(
        window_size=(1200, 900)
    )


    # -----------------------------------------
    # Add STL mesh
    # -----------------------------------------

    if geometry.mesh is not None:

        plotter.add_mesh(
            pv.wrap(geometry.mesh),
            color="lightgray",
            opacity=0.15,
            show_edges=False
        )


    # -----------------------------------------
    # Add blocks
    # -----------------------------------------

    for block in geometry.blocks:


        x0, y0, z0 = block.start
        x1, y1, z1 = block.end


        cube = pv.Box(
            bounds=(
                x0, x1,
                y0, y1,
                z0, z1
            )
        )


        # -------------------------
        # Substrate
        # -------------------------

        if block.is_substrate:


            # Fade interface region only
            if block.layer >= -3:

                plotter.add_mesh(
                    cube,
                    color="royalblue",
                    opacity=0.15,
                    show_edges=True,
                    edge_color="gray"
                )


            else:

                plotter.add_mesh(
                    cube,
                    color="royalblue",
                    opacity=0.45,
                    show_edges=True,
                    edge_color="black"
                )


        # -------------------------
        # Deposition
        # -------------------------

        else:


            plotter.add_mesh(
                cube,
                color="crimson",
                opacity=1.0,
                show_edges=True,
                edge_color="black"
            )



    # -----------------------------------------
    # Labels
    # -----------------------------------------

    plotter.add_axes()

    plotter.show_grid()

    plotter.add_title(
        "DED Block Discretization"
    )


    # Camera view
    plotter.view_isometric()


    plotter.show()



# ---------------------------------------------------
# Geometry coordinate check
# ---------------------------------------------------

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

    print(
        "xmin:",
        min(b.start[0] for b in substrate)
    )

    print(
        "xmax:",
        max(b.end[0] for b in substrate)
    )


    print(
        "ymin:",
        min(b.start[1] for b in substrate)
    )

    print(
        "ymax:",
        max(b.end[1] for b in substrate)
    )


    print(
        "zmin:",
        min(b.start[2] for b in substrate)
    )

    print(
        "zmax:",
        max(b.end[2] for b in substrate)
    )



    print("\nDeposition:")

    print(
        "xmin:",
        min(b.start[0] for b in deposition)
    )

    print(
        "xmax:",
        max(b.end[0] for b in deposition)
    )


    print(
        "ymin:",
        min(b.start[1] for b in deposition)
    )

    print(
        "ymax:",
        max(b.end[1] for b in deposition)
    )


    print(
        "zmin:",
        min(b.start[2] for b in deposition)
    )

    print(
        "zmax:",
        max(b.end[2] for b in deposition)
    )