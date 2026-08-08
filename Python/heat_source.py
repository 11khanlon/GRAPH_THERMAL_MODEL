"""
Step 5
Laser heat source model
Produces the initial nodal temperature field To before conduction begins
This module ONLY applies laser heating
--> No conduction
--> No convection
--> No graph operations

Implements Section 4.3.3
Goldak's double ellipsoid model (Paper Equation 19)

"""

import numpy as np

class GoldakHeatSource:

    def __init__(self,
                laser_power,
                conductivity,
                diffusivity,
                scan_speed,
                scaling_factor,
                meltpool_temperature,
                liquidus_temperature):

        self.P = laser_power
        self.k = conductivity
        self.alpha = diffusivity
        self.V = scan_speed
        self.C = scaling_factor

        self.Tmelt = meltpool_temperature
        self.Tliquidus = liquidus_temperature

    # ------------------------------------

    def goldak_temperature(self, x, y, z):

        """
        Equation (19)
        Returns laser-induced temperature at local coordinates (x,y,z)
        """

        r = np.sqrt(x**2 + y**2 + z**2)

        # avoid 0 in denomenator
        r = max(r, 1e-8)

        #Equation 19 
        temperature = (self.C * self.P
            / (2 * np.pi * self.k * r)
            * np.exp(
                -(self.V / (2 * self.alpha))
                * (x + r)
            )
        )

        return temperature

    # -------------------------------

    def heat_block(self,
                   nodes,
                   block_center,
                   temperature):
        
        """
        This function heats only the newly deposited block
        Returns updated temperature vector
        """

        T = temperature.copy()

        xc, yc, zc = block_center

        for node in block.nodes:

            index = node.id

            if not node.active:
                continue

            x = node.position[0] - xc
            y = node.position[1] - yc
            z = node.position[2] - zc

            delta = self.goldak_temperature(x, y, z)

            T[index] = max(T[index], min(delta, self.Tmelt))

        return T

    # ------------------------------------

    def heat_subsurface(self,
                        nodes,
                        block_center,
                        temperature,
                        cutoff = 0.20):
        
        """
        Heat underlying layers
        Paper considers reheating down to approximately
        20% of liquidus temperature
        """

        T = temperature.copy()

        threshold = (cutoff * self.Tliquidus)

        xc, yc, zc = block_center

        for i, node in enumerate(nodes):

            if not node.active:
                continue

            x = node.position[0] - xc
            y = node.position[1] - yc
            z = node.position[2] - zc

            delta = self.goldak_temperature(x, y, z)

            if delta >= threshold:

                T[i] = max(T[i], delta)

        return T

    # ------------------------------
    #Surface + subsurface heating

    def apply(self,
            nodes,
            block_center,
            temperature):
    
        T = self.heat_block(
            nodes,
            block_center,
            temperature)

        T = self.heat_subsurface(
            nodes,
            block_center,T)

        return T


# ---------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from config import LASER, MATERIAL, BUILD
    from material import Ti64Material   

    stl_file = BUILD["stl_file"]

    geom = Geometry(stl_file)

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    nodes = generator.nodes


    material = Ti64Material()
    
    heat = GoldakHeatSource(

        laser_power = LASER["power"],

        conductivity = Ti64Material.conductivity(),

        diffusivity = Ti64Material.diffusivity_linear_si(),

        scan_speed = LASER["scan_speed"],

        scaling_factor = LASER["goldak_C"],

        meltpool_temperature = LASER["meltpool_temperature"],

        liquidus_temperature = MATERIAL["liquidus_temperature"]
    )

    temperature = np.full(len(nodes), MATERIAL["ambient_temperature"])

    block = geom.blocks[0]

    center = block.center

    heated = heat.apply(
        nodes,
        center,
        temperature,
    )

    print("Maximum temperature:", heated.max())