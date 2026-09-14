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

class GoldakArcHeatSource:

    def __init__(self,
                conductivity,
                alpha, 
                scan_speed,
                scaling_factor,
                meltpool_temperature,
                liquidus_temperature, 
                waam):

        self.alpha = alpha
        self.k = conductivity
        self.V = scan_speed
        self.C = scaling_factor

        self.Tmelt = meltpool_temperature
        self.Tliquidus = liquidus_temperature


        #WAAM electrical parameters 
        self.voltage = waam["voltage"]                  # V
        self.current = waam["current"]                  # A
        self.thermal_efficiency = waam["thermal_efficiency"]

        #Calculate WAAM power once 
        self.power = self.waam_arc_power()



    #-------- WAAM Information ------
    def waam_arc_power(self):

        effective_power = self.thermal_efficiency * self.voltage * self.current

        return effective_power
    

    # --------------DED Powder Information -----------------

    def goldak_temperature(self, x, y, z):

        """
        Equation (19)
        Returns laser-induced temperature at local coordinates (x,y,z)
        """

        r = np.sqrt(x**2 + y**2 + z**2)

        # avoid 0 in denomenator
        r = max(r, 1e-8)

        #Equation 19 
        temperature = (self.power
            / (2 * np.pi * self.k * r)
            * np.exp(
                -(self.V / (2 * self.alpha))
                * (x + r)
            )
        )

        return temperature

    # -------------------------------

    def heat_block(self, block, temperature):
        
        """
        This function heats only the newly deposited block
        Returns updated temperature vector
        """

        xc, yc, zc = block.center
        
        T = temperature.copy()

        for node in block.nodes:

            if not node.active:
                continue

            index = node.id

            x = node.position[0] - xc
            y = node.position[1] - yc
            z = node.position[2] - zc

            delta =  self.goldak_temperature(x, y, z)

            T[index] = max(T[index], min(delta, self.Tmelt))

        return T

    # ------------------------------------

    def heat_subsurface(self,
                        nodes,
                        block,
                        temperature,
                        cutoff = 0.20):
        
        """
        Heat underlying layers
        Paper considers reheating down to approximately
        20% of liquidus temperature
        """

        T = temperature.copy()

        threshold = (cutoff * self.Tliquidus)

        xc, yc, zc = block.center

        for i, node in enumerate(nodes):

            if not node.active:
                continue

            x = node.position[0] - xc
            y = node.position[1] - yc
            z = node.position[2] - zc

            delta =  self.goldak_temperature(x, y, z)

            if delta >= threshold:

                T[i] = max(T[i], min(delta, self.Tmelt))

        return T

    # ------------------------------
    #Surface + subsurface heating

    def apply(self, nodes, block, temperature):
    
        T = self.heat_block(block, temperature)

        T = self.heat_subsurface(nodes, block, T)

        return T


# ---------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from config import LASER, MATERIAL, BUILD, WAAM
 

    stl_file = BUILD["stl_file"]

    geom = Geometry(stl_file)

    geom.build_blocks()

    generator = NodeGenerator(geom)

    generator.generate()

    nodes = generator.nodes

    heat = GoldakArcHeatSource(


        conductivity = 6.8,  #temporary, just for test case, 
            
        alpha = 2.7228E-6,  #temporary, just for test case, 

        scan_speed = LASER["scan_speed"],

        scaling_factor = LASER["goldak_C"],

        meltpool_temperature = LASER["meltpool_temperature"],

        liquidus_temperature = MATERIAL["liquidus_temperature"], 

        waam = WAAM
    )


    temperature = np.full(len(nodes), MATERIAL["ambient_temperature"])

    #block = geom.blocks[2560]
    block = geom.blocks[0]
    #block = next(block for block in geom.blocks  if not block.is_substrate)

    heated = heat.apply(nodes, block, temperature)


    print("\n----------- Goldak Verification -----------")

    print(f"Block ID        : {block.id}")
    print(f"Layer           : {block.layer}")
    print(f"Block center    : {block.center}")

    print()

    print(f"Ambient Temp    : {MATERIAL['ambient_temperature']:.1f} °C")
    print(f"Maximum Temp    : {heated.max():.1f} °C")
    print(f"Minimum Temp    : {heated.min():.1f} °C")

    heated_nodes = np.sum(
        heated > MATERIAL["ambient_temperature"]
    )

    print(f"Total Heated Nodes    : {heated_nodes}")

    print("-------------------------------------------")

    changed = 0

    for node in block.nodes:

        if heated[node.id] > MATERIAL["ambient_temperature"]:
            changed += 1

    print(f"Nodes in current block        : {len(block.nodes)}")
    print(f"Nodes actually heated in current block : {changed}")



    print("\n--------- Nodes in current block: ----------")

    for node in block.nodes:

        x = node.position[0] - block.center[0]
        y = node.position[1] - block.center[1]
        z = node.position[2] - block.center[2]

        r = np.sqrt(x**2 + y**2 + z**2)

        delta_T = heat.goldak_temperature(x, y, z)

        print(
            f"Node {node.id}: "
            f"position={node.position}, "
            f"local=({x:.6e}, {y:.6e}, {z:.6e}), "
            f"r={r:.6e}, "
            f"deltaT={delta_T:.2f}"
        )

    print("--------------------------------------")
