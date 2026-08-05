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
        

        beta = self.beta(h, cp)

        T = temperature.copy()

        factor = np.exp(-beta * dt)

        boundary = np.where(surface_mask)[0]

        T[boundary] = (ambient_temperature 
                       + ( T[boundary] - ambient_temperature) 
                       * factor)

        return T

    # -------------------------------------------------------
    #Cooling immediately after one block
    def block_cooling(self,
                    temperature,
                    surface_mask,
                    h,
                    cp,
                    ambient_temperature,
                    block_time):
        
        return self.cool(
            temperature,
            surface_mask,
            h,
            cp,
            ambient_temperature,
            block_time)

    # -------------------------------------------------------
    #cooling during dwell time
    def dwell_cooling(self,
                    temperature,
                    surface_mask,
                    h,
                    cp,
                    ambient_temperature,
                    dt = 1.0):

        return self.cool(temperature,
                        surface_mask,
                        h,
                        cp,
                        ambient_temperature,
                        dt)

    # -------------------------------------------------------

    def multiple_steps(self, 
                       temperature,
                       surface_mask,
                       h,
                       cp,
                       ambient_temperature,
                       total_time,
                       dt = 1.0):
        """
        Repeated convection cooling.
        Used for arbitrary dwell periods.
        """

        T = temperature.copy()

        steps = int(np.ceil(total_time / dt))

        for _ in range(steps):
            T = self.cool(T,
                          surface_mask,
                          h,
                          cp,
                          ambient_temperature,
                          dt)

        return T


# ------------------------------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from config import MATERIAL, CONVECTION, BLOCK
    from material import Ti64Material

    geom = Geometry("example_part.stl")

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    nodes = generator.nodes

    temperature = np.full(
        len(nodes),
        MATERIAL["ambient_temperature"])

    temperature[:50] = 1200.0

    surface = np.array(
        [node.surface for node in nodes],
        dtype=bool)

    solver = ConvectionSolver(
        density = MATERIAL["density"],
        block_length = BLOCK["block_length"])

    cooled = solver.block_cooling(
        temperature,
        surface,
        h = CONVECTION["forced"],
        cp = material.specific_heat(np.mean(temperature),
        ambient_temperature = MATERIAL["ambient_temperature"],
        block_time = BLOCK["time_per_block"])

    print("Maximum temperature:", np.max(cooled))