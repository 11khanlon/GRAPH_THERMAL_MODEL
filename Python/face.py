import numpy as np



#--------- FACE-TOUCHING TEST -----------

def faces_touch(block_a, block_b, tolerance=1e-10):

    ax0, ay0, az0 = block_a.start
    ax1, ay1, az1 = block_a.end

    bx0, by0, bz0 = block_b.start
    bx1, by1, bz1 = block_b.end

    touching = set()

    
    # X direction
    y_overlap = min(ay1, by1) - max(ay0, by0)
    z_overlap = min(az1, bz1) - max(az0, bz0)

    if (
        abs(ax1 - bx0) < tolerance
        and y_overlap > tolerance
        and z_overlap > tolerance
    ):
        touching.add("right")

    if (
        abs(ax0 - bx1) < tolerance
        and y_overlap > tolerance
        and z_overlap > tolerance
    ):
        touching.add("left")


    # Y direction
    x_overlap = min(ax1, bx1) - max(ax0, bx0)

    if (
        abs(ay1 - by0) < tolerance
        and x_overlap > tolerance
        and z_overlap > tolerance
    ):
        touching.add("front")

    if (
        abs(ay0 - by1) < tolerance
        and x_overlap > tolerance
        and z_overlap > tolerance
    ):
        touching.add("back")

    #Z Direction

    if (
        abs(az1 - bz0) < tolerance
        and x_overlap > tolerance
        and y_overlap > tolerance
    ):
        touching.add("top")

    if (
        abs(az0 - bz1) < tolerance
        and x_overlap > tolerance
        and y_overlap > tolerance
    ):
        touching.add("bottom")

    return touching


# ---------------------------------------------------------
# UPDATE EXPOSED FACES

def update_exposed_faces(geometry):

    active_blocks = [
        block
        for block in geometry.blocks
        if block.active
    ]

    # Start by assuming every face is exposed

    for block in active_blocks:

        for face in block.exposed_faces:

            block.exposed_faces[face] = True


    # Find faces covered by another active block

    opposite = {
        "left": "right",
        "right": "left",
        "front": "back",
        "back": "front",
        "top": "bottom",
        "bottom": "top"
    }

    for i, block_a in enumerate(active_blocks):

        for block_b in active_blocks[i + 1:]:

            touching_faces = faces_touch(
                block_a,
                block_b
            )

            for face_a in touching_faces:

                block_a.exposed_faces[face_a] = False

                face_b = opposite[face_a]

                block_b.exposed_faces[face_b] = False


# ---------------------------------------------------------
# ACTIVATE BLOCK

def activate_block(geometry, block):

    """
    Activate a block and update the exposed
    surfaces of the active geometry.
    """

    block.active = True

    update_exposed_faces(geometry)


# ---------------------------------------------------------
# ACTIVE BLOCKS

def active_blocks(geometry):

    return [
        block
        for block in geometry.blocks
        if block.active
    ]


# ---------------------------------------------------------
# PRINT EXPOSED FACES

def print_exposed_faces(geometry):

    print("\n--------- EXPOSED FACES ---------")

    for block in active_blocks(geometry):

        exposed = [
            face
            for face, is_exposed
            in block.exposed_faces.items()
            if is_exposed
        ]

        print(
            f"Block {block.id} "
            f"(layer={block.layer}) : "
            f"{exposed}"
        )

    print("---------------------------------")

def print_exposure_statistics(geometry):
    print("\n------ EXPOSURE STATISTICS ------")

    for face in [
        "top",
        "bottom",
        "left",
        "right",
        "front",
        "back"
    ]:
        count = sum(
            block.exposed_faces[face]
            for block in active_blocks(geometry)
        )

        print(f"{face:>7}: {count}")

    print(
        "Total active blocks:",
        len(active_blocks(geometry))
    )

    print("---------------------------------")