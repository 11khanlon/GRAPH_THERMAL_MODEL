"""
Heat loss through convection (and radiation approximation)

Paper Equations:
(16): Tb = Tc exp(-β tb)
(18): TLf = TLc exp(-β t)

"""

import numpy as np

class ConvectionSolver:

    #Newton cooling model 
    def __init__(self,
                 density,
                 block_length):
        
        self.rho = density
        self.L = block_length
    

    # -------------------------------------------------------

    def beta(self, h, cp):

        #β = h / (rho L Cp)

        return h / (self.rho * self.L * cp )

    # -------------------------------------------------------
    #Apply Newton cooling to surface nodes
    def cool(self,
            temperature,
            surface_mask,
            h,
            cp,
            ambient_temperature,
            dt):


        T = temperature.copy()

        beta = self.beta(h, cp)

        factor = np.exp(-beta * dt)

        T[surface_mask] = (
            ambient_temperature
            + (T[surface_mask] - ambient_temperature)
            * factor[surface_mask]
        )

        return T

    
    # -------------------------------------------------------
    #Cooling immediately after one block
    def block_cooling(
            self,
            temperature,
            surface_mask,
            h,
            cp,
            ambient_temperature,
            block_time
        ):

        return self.cool(
            temperature,
            surface_mask, 
            h, 
            cp, 
            ambient_temperature,
            block_time
        )

    # -------------------------------------------------------
    #cooling during dwell time
    def dwell_cooling(
                self,
                temperature,
                surface_mask,
                h,
                cp,
                ambient_temperature,
                dt):
    

        return self.cool(
            temperature,
            surface_mask, 
            h, 
            cp, 
            ambient_temperature, 
            dt)



# ------------------------------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from config import MATERIAL, CONVECTION, BLOCK, BUILD
    from material import Ti64Material
    from face import update_exposed_faces

    surface_band_thickness = BLOCK["surface_band_thickness"]

    surface_thickness = BLOCK["surface_thickness"] 

    stl_file = BUILD["stl_file"]
        
    geom = Geometry(stl_file)

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    nodes = generator.nodes

    update_exposed_faces(geom)

    generator.update_node_exposure(surface_thickness)

    generator.update_surface_types()

    #Create surface masks 
    forced_surface_mask = np.array(
        [node.forced_surface for node in nodes],
        dtype=bool)

    free_surface_mask = np.array(
        [node.free_surface for node in nodes],
        dtype=bool)

    #initial temperature 
    temperature = np.full(
        len(nodes),
        MATERIAL["ambient_temperature"])

    temperature[forced_surface_mask] = 1200.0

    solver = ConvectionSolver(
        density = MATERIAL["density"],
        block_length = BLOCK["length"])

    material = Ti64Material() 
    cp = material.specific_heat(temperature)

    cooled_forced = solver.block_cooling(
        temperature=temperature,
        surface_mask=forced_surface_mask,
        h=CONVECTION["forced"],
        cp=cp,
        ambient_temperature=MATERIAL["ambient_temperature"],
        block_time=BLOCK["time_per_block"]
    )


    cooled_free = solver.block_cooling(
        temperature=cooled_forced,
        surface_mask=free_surface_mask,
        h=CONVECTION["free"],
        cp=cp,
        ambient_temperature=MATERIAL["ambient_temperature"],
        block_time=BLOCK["time_per_block"]
    )

    print("\n------ CONVECTION TEST ------")

    print(
        f"Forced surface nodes : "
        f"{np.sum(forced_surface_mask)}"
    )

    print(
        f"Free surface nodes   : "
        f"{np.sum(free_surface_mask)}"
    )

    print(
    "Maximum forced-surface temperature before:",
    np.max(temperature[forced_surface_mask])
    )

    print(
        "Maximum forced-surface temperature after:",
        np.max(cooled_forced[forced_surface_mask])
    )

    print("\n------ SURFACE MASK TEST ------")

    print("Total nodes:", len(nodes))

    print("Forced mask shape:", forced_surface_mask.shape)
    print("Free mask shape:", free_surface_mask.shape)

    print("Forced surface nodes:", np.sum(forced_surface_mask))
    print("Free surface nodes:", np.sum(free_surface_mask))

    print("Forced mask dtype:", forced_surface_mask.dtype)
    print("Free mask dtype:", free_surface_mask.dtype)