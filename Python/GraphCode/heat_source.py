"""
Step 5
Laser heat source model
Produces the initial nodal temperature field To before conduction begins
This module ONLY applies laser heating
--> No conduction
--> No convection
--> No graph operations

Implements Section 4.3.3 of paper and modified for Arc 
"""

import numpy as np

class ArcHeatSource:

    def __init__(
        self,
        material_model,
        reference_temperature,
        liquidus_temperature,
        waam):

        # Material model
        self.material = material_model

        # Constant-property reference temperature
        # for Rosenthal source equation
        self.Tref = reference_temperature

        # Look up material properties at Tref
        self.k = self.material.conductivity(self.Tref)
        self.cp = self.material.specific_heat(self.Tref)
        self.alpha = self.material.diffusivity(self.Tref)

        self.Tliquidus = liquidus_temperature

        # WAAM process parameters
        self.V = waam["scan_speed"]
        self.voltage = waam["voltage"]
        self.current = waam["current"]
        self.thermal_efficiency = waam["thermal_efficiency"]

        # Constant effective arc power
        self.power = self.waam_arc_power()

        print("\n------ ARC SOURCE MATERIAL PROPERTIES ------")
        print(f"Reference temperature: {self.Tref:.2f} °C")
        print(f"Thermal conductivity:  {self.k:.4f} W/m-K")
        print(f"Specific heat:         {self.cp:.2f} J/kg-K")
        print(f"Thermal diffusivity:   {self.alpha:.6e} m²/s")
        print(f"Effective arc power:   {self.power:.2f} W")



    #-------- WAAM Information ------
    def waam_arc_power(self):

        effective_power = self.thermal_efficiency * self.voltage * self.current

        return effective_power
    

    # --------------DED Powder Information -----------------

    def arc_temperature_rise(self, x, y, z):

        #Returns laser-induced temperature at local coordinates (x,y,z)
  
        r = np.sqrt(x**2 + y**2 + z**2)

        # avoid 0 in denomenator
        r = max(r, 1e-8)

        #Equation 19 
        delta_T = (self.power
            / (2 * np.pi * self.k * r)
            * np.exp(
                -(self.V / (2 * self.alpha))
                * (x + r)
            )
        )

        return delta_T

    # -------------------------------

    def heat_block(self, nodes, block):
        
        """
        This function heats only the newly deposited block
        Returns updated temperature vector
        """

        delta_T = np.zeros(len(nodes), dtype = float)

        xc, yc, zc = block.center

        for node in block.nodes:

            if not node.active:
                continue

            x = node.position[0] - xc
            y = node.position[1] - yc
            z = node.position[2] - zc

            delta_T[node.id] = (self.arc_temperature_rise(x, y, z))

        return delta_T

    # ------------------------------------

    def heat_subsurface(self, nodes, block, temperature, cutoff = 0.20):

        # Calculate arc-induced temperature rise for previously existing active material within the subsurface heating threshold
        
        delta_T = np.zeros(len(nodes), dtype=float)

        threshold = cutoff * self.Tliquidus
        
        xc, yc, zc = block.center

        for node in nodes:

            if not node.active:
                continue

            # Current block is already handled by heat_block()
            if node.block_id == block.id:
                continue

            x = node.position[0] - xc
            y = node.position[1] - yc
            z = node.position[2] - zc

            node_delta_T = self.arc_temperature_rise(x, y, z)

            predicted_temperature = temperature[node.id] + node_delta_T
            
            if predicted_temperature >= threshold:

                delta_T[node.id] = node_delta_T
                

        return delta_T

    # ------------------------------
    #Surface + subsurface heating

    def apply(self, nodes, block, temperature):

        delta_T_block = self.heat_block(nodes, block)

        delta_T_subsurface = self.heat_subsurface(nodes, block, temperature)
        
        delta_T_arc = delta_T_block + delta_T_subsurface
        
        return delta_T_arc


# ---------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from config import  MATERIAL, BUILD, WAAM
    from material import Al5183Material, Al6061Material
        
    stl_file = BUILD["stl_file"]
    substrate_material = Al6061Material()
    wire_material = Al5183Material()
    geom = Geometry(stl_file, substrate_material=substrate_material, wire_material=wire_material)

    geom.build_blocks()
    generator = NodeGenerator(geom)
    generator.generate()
    nodes = generator.nodes


    block = next(
        block
        for block in geom.blocks
        if not block.is_substrate
    )

    material_model = block.material

    heat = ArcHeatSource(
        material_model=material_model,
        reference_temperature=MATERIAL["ambient_temperature"],
        liquidus_temperature=MATERIAL["liquidus_temperature"],
        waam=WAAM
    )

    heat = ArcHeatSource(
        material_model = material_model,
        reference_temperature = MATERIAL["ambient_temperature"],
        liquidus_temperature = MATERIAL["liquidus_temperature"],
        waam = WAAM)


    temperature = np.full(len(nodes), MATERIAL["ambient_temperature"])

    block = next(
        block
        for block in geom.blocks
        if not block.is_substrate )

    block.active = True

    for node in block.nodes:
        node.active = True

    delta_T_arc = heat.apply(nodes, block, temperature)
    heated = temperature + delta_T_arc


    print("\n----------- Arc Heating Verification -----------")

    print(f"Block ID        : {block.id}")
    print(f"Layer           : {block.layer}")
    print(f"Block center    : {block.center}")

    print()

    print(f"Ambient Temp    : {MATERIAL['ambient_temperature']:.1f} °C")
    print(f"Maximum Temp    : {heated.max():.1f} °C")
    print(f"Minimum Temp    : {heated.min():.1f} °C")

    heated_nodes = np.sum(heated > MATERIAL["ambient_temperature"])

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

        delta_T = heat.arc_temperature_rise(x, y, z)

        print(
            f"Node {node.id}: "
            f"position={node.position}, "
            f"local=({x:.6e}, {y:.6e}, {z:.6e}), "
            f"r={r:.6e}, "
            f"deltaT={delta_T:.2f}"
        )

    print("--------------------------------------")
