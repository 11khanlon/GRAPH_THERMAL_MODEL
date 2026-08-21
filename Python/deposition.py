# Main deposition simulation

import numpy as np

from conduction import ConductionSolver
from convection import ConvectionSolver
from heat_source import GoldakHeatSource
from material import Ti64Material

from face import update_exposed_faces

#------------------------
class DepositionSimulation:

    def __init__(self,
                geometry,
                node_generator,
                graph,
                material,
                laser,
                convection,
                graph_settings,
                dwell_info, 
                block_info):

        self.geometry = geometry
        self.node_generator = node_generator
        self.nodes = node_generator.nodes
        self.graph = graph
        self.dwell_info = dwell_info
        self.block_info = block_info

        #----- Configuration libraries ----------
        self.material_settings = material
        self.laser = laser
        self.convection = convection
        self.graph_settings = graph_settings

        #------ Temperature dependent material model ---------
        self.material_model = Ti64Material()

        #-------- Surface band definition -----------
        self.surface_thickness = block_info["surface_thickness"]

        #------- Initial temperature state ------------
        self.temperature = np.full(len(self.nodes),
                                   material["ambient_temperature"],
                                   dtype = float)

        #---------- Constants concerning time --------------
        self.block_time = block_info["time_per_block"]    #Process timing
        self.dwell_dt = dwell_info["time_step"]           # Paper uses 1-second dwell integration
        self.dwell_time = dwell_info["case_A"]            # Total dwell between layers
        self.time = 0.0                              #simulation time 

        #--------- Initial state as history -------------
        self.history = [self.temperature.copy()]
        self.history_time = [self.time]

        #------- Solvers ---------- 
        self.conduction = ConductionSolver(graph,
            gain = graph_settings["gain"])

        self.cooling = ConvectionSolver(
            density = material["density"],
            block_length = block_info["block_length"])

        self.heat_source = GoldakHeatSource(
            laser_power =  laser["power"],
            conductivity = laser["thermal_conductivity"],
            diffusivity = laser["thermal_diffusivity"],
            scan_speed = laser["scan_speed"],
            scaling_factor = laser["goldak_C"],
            meltpool_temperature = laser["meltpool_temperature"],
            liquidus_temperature = material["liquidus_temperature"])

    # -----------------------------------------------------
    #activate nodes belonging to the newly deposited block 
    def activate_block(self, block):

        block.active = True

        for node in block.nodes:
            node.active = True

    #-----------------------------------------
    def update_surface_masks(self):

        self.forced_surface_mask = np.array(
            [node.forced_surface for node in self.nodes],
            dtype=bool)

        self.free_surface_mask = np.array(
            [node.free_surface for node in self.nodes],
            dtype=bool)

        print(f"Forced surface nodes: " 
              f"{np.sum(self.forced_surface_mask)}")

        print(f"Free surface nodes: "
            f"{np.sum(self.free_surface_mask)}")

    # --------------------------------
    def active_mask(self):

        return np.array(
            [node.active for node in self.nodes],
            dtype=bool
        )

    #------------------------------------
    def layer_indices(self, layer):

        return np.array([
            node.id
            for node in self.nodes
            if (
                node.active
                and not node.is_substrate
                and node.block.layer == layer
            )
        ])
    #---------------------------------------
    def diffusivity_from_layer(self, layer):

        indices = self.layer_indices(layer)

        if len(indices) == 0:
            return self.material_model.diffusivity_linear_si(
                self.material_settings["ambient_temperature"]
            )

        T_layer = self.temperature[indices]

        return self.material_model.layer_diffusivity(
            T_layer
        )
    
    #-----------------------------------------
    def record_state(self):

        self.history.append(
            self.temperature.copy()
        )

        self.history_time.append(self.time)
  
    # -----------------------------------------------

    def simulate_block(self, block):

        self.activate_block(block)

        # Update geometry-level exposed faces
        update_exposed_faces(self.geometry)

        # Convert block faces → exposed nodes
        self.node_generator.update_node_exposure(self.surface_thickness)

        # Classify nodes as forced/free
        self.node_generator.update_surface_types()

        # Rebuild masks
        self.update_surface_masks()

        # Step 1 -> Laser heating
        self.temperature = self.heat_source.apply(
            self.nodes,
            block,
            self.temperature)

  
        # Step 2 -> Conduction
        alpha = self.material.layer_diffusivity(self.temperature)

        self.temperature = (
            self.conduction.block_conduction(
            self.temperature,
            alpha,
            self.block_info["time_per_block"]
                )
            )

        # Step 3 --> Convection
        cp = self.material.specific_heat(self.temperature)

        overlap = (
            self.forced_surface_mask
            & self.free_surface_mask
        )

        if np.any(overlap):
            print(
                "WARNING: forced/free overlap:",
                np.sum(overlap)
            )

        if self.node.is_substrate:

            if "top" in self.node.exposed_faces:

                self.node.forced_surface = True
                self.node.free_surface = False

            elif any(
                face in self.node.exposed_faces
                for face in [
                    "left",
                    "right",
                    "front",
                    "back",
                    "bottom"
                ]
            ):

                self.node.free_surface = True

        # Forced convection + equivalent radiation
        self.temperature = self.cooling.block_cooling(
            temperature=self.temperature,
            surface_mask=self.forced_surface_mask,
            h=self.convection["forced"],
            cp=cp,
            ambient_temperature=
                self.material_settings["ambient_temperature"],
            block_time=
                self.block_info["time_per_block"]
        )

        # Free convection
        self.temperature = self.cooling.block_cooling(
            temperature=self.temperature,
            surface_mask=self.free_surface_mask,
            h=self.convection["free"],
            cp=cp,
            ambient_temperature=
                self.material_settings["ambient_temperature"],
            block_time=
                self.block_info["time_per_block"]
        )

        self.time += self.block_time
        self.record_state()
        self.history.append(self.temperature.copy())

    # ----------------------------------
    def simulate_dwell(self, dwell_time):

        steps = int(dwell_time)

        active_mask = np.array(
            [node.active for node in self.nodes],
            dtype=bool)

        for _ in range(steps):

            alpha = (self.material_model.layer_diffusivity(
                    self.temperature[active_mask]
                )
            )

            self.temperature = (
                self.conduction.dwell_conduction(
                    temperature=self.temperature,
                    alpha=alpha,
                    dt= self.dwell_info["time_step"],
                    active_mask=active_mask
                )
            )

            cp = self.material_model.specific_heat(
                self.temperature
            )

            self.temperature = (
                self.cooling.dwell_cooling(
                    temperature=self.temperature,
                    surface_mask=
                        self.forced_surface_mask,
                    h=self.convection["forced"],
                    cp=cp,
                    ambient_temperature=
                        self.material_settings[
                            "ambient_temperature"],
                    dt=self.dwell_info["time_step"]
                )
            )

            cp = self.material_model.specific_heat(
                self.temperature
            )

            self.temperature = (
                self.cooling.dwell_cooling(
                    temperature=self.temperature,
                    surface_mask=
                        self.free_surface_mask,
                    h=self.convection["free"],
                    cp=cp,
                    ambient_temperature=
                        self.material_settings[
                            "ambient_temperature"
                        ],
                    dt=self.dwell_info["time_step"]
                )
            )

            self.history.append(
                self.temperature.copy()
            )

    
    # --------------------------------------
    def run(self):

        print("----------------------------")
        print("Beginning deposition")
      

        current_layer = None

        ordered_blocks = self.geometry.deposition_order()

        for block in ordered_blocks:

            if block.layer != current_layer:

                # Dwell AFTER previous layer
                if current_layer is not None:

                    print(
                        f"Dwell after layer "
                        f"{current_layer + 1}"
                    )

                    self.simulate_dwell()

                current_layer = block.layer

                print(
                    f"Layer {current_layer + 1}"
                )

            # Deposit current block
            self.simulate_block(block)

        print("-------------------")
        print("Simulation complete!")

        self.history = np.asarray(
            self.history
        )

        self.history_time = np.asarray(
            self.history_time
        )

        return self.history

    # --------------------------------------

    def sensor_history(self,
        sensor_index):

        return self.history[:, sensor_index]

    # --------------------------------------

    def maximum_temperature(self):

        return np.max(self.temperature)

    # ------------------------------

    def final_temperature(self):

        return self.temperature


# ------------------------------------

if __name__ == "__main__":

    from geometry import Geometry
    from nodes import NodeGenerator
    from graph import ThermalGraph
    from config import (MATERIAL, LASER, GRAPH, CONVECTION, BLOCK, DWELL, BUILD)

    stl_file = BUILD["stl_file"]

    geometry = Geometry(stl_file)

    geometry.build_blocks()

    generator = NodeGenerator(geometry
    )

    generator.generate()

    graph = ThermalGraph(
        generator.nodes
    )

    graph.build()
    graph.degree_matrix()
    graph.laplacian()
    graph.eigensystem()

    # Choose experimental case here
    dwell_time = DWELL["case_A"]

    simulation = DepositionSimulation(
        geometry=geometry,
        node_generator=generator,
        graph=graph,
        material=MATERIAL,
        laser=LASER,
        convection=CONVECTION,
        graph_settings=GRAPH,
        dwell_info = DWELL, 
        block_info = BLOCK
    )

    history = simulation.run()

    print(
        "Maximum temperature:",
        simulation.maximum_temperature()
    )

    print(
        "Stored states:",
        history.shape[0]
    )