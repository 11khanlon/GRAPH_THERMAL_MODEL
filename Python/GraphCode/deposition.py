# Main deposition simulation

import numpy as np

from conduction import ConductionSolver
from convection import ConvectionSolver
from heat_source import ArcHeatSource
from material import Al5183Material, Al6061Material
from datetime import datetime
import time 
from pathlib import Path
from face import update_exposed_faces
from deposition_tracker import DepositionTracker


#------------------------
class DepositionSimulation:

    def __init__(self,
                geometry,
                node_generator,
                graph,
                material,
                convection,
                graph_settings,
                dwell_info,
                dwell_time, 
                block_info, 
                build_info,
                waam,
                probes = None):

        self.geometry = geometry
        self.node_generator = node_generator
        self.nodes = node_generator.nodes
        self.graph = graph
        self.dwell_info = dwell_info
        self.dwell_time = dwell_time
        self.block_info = block_info
        self.build_info = build_info 
        self.waam = waam 


        #----- Configuration libraries ----------
        self.material_settings = material
        self.convection = convection
        self.graph_settings = graph_settings

        #------ Temperature dependent material model ---------
        self.substrate_material = geometry.substrate_material
        self.wire_material = geometry.wire_material


        #-------- Surface band definition -----------
        self.surface_thickness = block_info["surface_thickness"]

        #------- Initial temperature state ------------
        self.temperature = np.full(len(self.nodes), material["ambient_temperature"], dtype = float)

        #----------- Block and Build Info -----------
        self.blocks_per_layer = build_info["blocks_per_layer"]
        self.block_length = self.geometry.length / self.blocks_per_layer
        

        #---------- Constants concerning time --------------
        self.block_time = self.block_length / self.waam["scan_speed"]    #Process timing
        self.dwell_dt = dwell_info["dwell_time_step"]           # Paper uses 1-second dwell integration
        self.time = 0.0                              #simulation time 

        #------- Solvers ---------- 
        self.conduction = ConductionSolver(graph,
            gain = graph_settings["gain"])

        rho = np.array([node.material.rho for node in self.nodes], dtype=float)

        self.cooling = ConvectionSolver(
            density = rho,
            block_length = self.block_length)

        self.heat_source = ArcHeatSource(
            material_model=self.wire_material,
            reference_temperature=self.material_settings["ambient_temperature"],
            liquidus_temperature=self.material_settings["liquidus_temperature"],
            waam=self.waam)

        #----------- Simulation metadata ------------
        self.wall_start = None
        self.wall_end = None
        self.computation_time = None
        self.tracker = DepositionTracker(nodes = self.nodes, initial_temperature = self.temperature, probes = probes)


    #------------- Helper Functions -----------
    #activate nodes belonging to the newly deposited block 
    def activate_block(self, block):

        block.active = True

        for node in block.nodes:
            node.active = True

    def update_surface_masks(self):

        self.forced_surface_mask = np.array(
            [node.forced_surface for node in self.nodes],
            dtype=bool)

        self.free_surface_mask = np.array(
            [node.free_surface for node in self.nodes],
            dtype=bool)

        print(f"Forced surface nodes: {np.sum(self.forced_surface_mask)} ")
        print(f"Free surface nodes: {np.sum(self.free_surface_mask)}")


    def deposition_temperature_history(self):

        deposition_indices = np.array([
            node.id
            for node in self.nodes
            if not node.is_substrate
        ], dtype=int)

        return np.asarray(self.history)[:, deposition_indices]
   

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
            return self.material_model.diffusivity(
                self.material_settings["ambient_temperature"]
            )

        T_layer = self.temperature[indices]

        return self.material_model.layer_diffusivity(T_layer)

    #--------------------------------------------------
    def current_diffusivity(self):

        alpha = np.array([
            node.material.diffusivity(self.temperature[i])
            for i, node in enumerate(self.nodes)
        ], dtype=float)

        return alpha

    #--------------------------------------------
    def active_mask(self):

        return np.array(
            [node.active for node in self.nodes],
            dtype=bool)

    #--------------------------------------------------
    def inactive_node_statistics(self):

        active = self.active_mask()
        inactive = ~active

        if not np.any(inactive):
            return

        Tinactive = self.temperature[inactive]

        ambient = self.material_settings["ambient_temperature"]

        print("Max inactive T:", Tinactive.max())
        print("Inactive nodes > ambient + 10 C:", np.sum(Tinactive > ambient + 10.0))

    #------------------------------------------------
    def hottest_inactive_node(self):

        active = self.active_mask()
        inactive = ~active

        if not np.any(inactive):
            return

        inactive_indices = np.where(inactive)[0]
        local_index = np.argmax(self.temperature[inactive])
        node_index = inactive_indices[ local_index]
        node = self.nodes[node_index]

        print(f"Hottest inactive node: {node.id} ")
        print(f"Temperature: {self.temperature[node_index]:.2f} °C ")
        print(f"Layer: {node.block.layer} ")
        print(f"Position: {node.position} ")

       
    #----------------------------------------
    def print_current_block_temperature(self, block, label):

        indices = np.array([
            node.id
            for node in block.nodes])

        Tblock = self.temperature[indices]

        print(
            f"{label} | "
            f"Block {block.id}: "
            f"Tmin={Tblock.min():.2f} °C, "
            f"Tmax={Tblock.max():.2f} °C")


    def check_temperature_validity(temperature, ambient_temperature):

        if not np.all(np.isfinite(temperature)):
            raise ValueError("Temperature contains NaN or infinity.")

        if np.any(temperature < ambient_temperature - 1e-6):
            print("WARNING: Some nodes are below ambient temperature.")

        print(f"Temperature range: {temperature.min():.2f} to  {temperature.max():.2f} °C")
         

    def check_inactive_heating(nodes, temperature, ambient_temperature, tolerance=1.0):

        inactive = np.array([
            not node.active
            for node in nodes])

        if not np.any(inactive):
            return

        inactive_temperatures = temperature[inactive]
        heated = np.sum(inactive_temperatures > ambient_temperature + tolerance)

        print(f"Inactive nodes above ambient: {heated}")

        if heated > 0:
            print("WARNING: Heat may be reaching inactive nodes.")


    def block_cooling_rate(block, previous_temperature, current_temperature, dt):

        if dt <= 0:
            raise ValueError("dt must be positive.")

        indices = np.array([
            node.id
            for node in block.nodes])

        T_old = np.mean(previous_temperature[indices])
        T_new = np.mean(current_temperature[indices])

        cooling_rate = (T_new - T_old) / dt
        print(f"Block {block.id} temperature rate: {cooling_rate:.2f} °C/s ")

        return cooling_rate
    
    # -----------------------------------------------
    def simulate_block(self, block):

        print("Temperature range:", self.temperature.min(), self.temperature.max())
        print("Active nodes:", np.sum(self.active_mask()))
        print("Inactive nodes:", np.sum(~self.active_mask()))

        self.activate_block(block)

        update_exposed_faces(self.geometry)                              # Update geometry-level exposed faces
        self.node_generator.update_node_exposure(self.surface_thickness)  # Convert block faces → exposed nodes
        self.node_generator.update_surface_types()                        # Classify nodes as forced/free
        self.update_surface_masks()                                        # Rebuild masks

        # Step 1: Arc/HeatSource heating
        delta_T_arc = self.heat_source.apply(nodes=self.nodes, block=block, temperature=self.temperature)
        self.temperature += delta_T_arc


        self.print_current_block_temperature(block, "After HeatSource")

        imax = np.argmax(self.temperature)
        hot_node = self.nodes[imax]

        print(
            f"Hottest node after heat source: "
            f"id={hot_node.id}, "
            f"block={hot_node.block_id}, "
            f"layer={hot_node.block.layer}, "
            f"T={self.temperature[imax]:.2f} °C, "
            f"position={hot_node.position}")

        self.tracker.record_probe_state(
            temperature=self.temperature,
            time=self.time,
            stage="after_heatsource",
            block=block)


        self.tracker.print_temperature_status(
            temperature=self.temperature,
            time=self.time,
            label="After HeatSource",
            block=block)

  
        # Step 2 -> Conduction
        alpha = self.current_diffusivity()

        self.temperature = (
            self.conduction.block_conduction(
            self.temperature,
            alpha,
            block_time = self.block_time)
            )
        
        self.tracker.record_probe_state(
            temperature=self.temperature,
            time=self.time,
            stage="after_conduction",
            block=block)
        
        self.tracker.print_temperature_status(
            temperature=self.temperature,
            time=self.time,
            label="After conduction",
            block=block)

        # Step 3 --> Convection
        cp = np.array([
            node.material.specific_heat(self.temperature[node.id])
            for node in self.nodes])

        overlap = (self.forced_surface_mask & self.free_surface_mask)

        if np.any(overlap):
            print("WARNING: forced/free overlap:", np.sum(overlap))


        # Forced convection + equivalent radiation
        self.temperature = self.cooling.block_cooling(
            temperature=self.temperature,
            surface_mask=self.forced_surface_mask,
            h=self.convection["forced"],
            cp=cp,
            ambient_temperature = self.material_settings["ambient_temperature"],
            block_time = self.block_time
        )

        cp = np.array([
            node.material.specific_heat(self.temperature[i])
            for i, node in enumerate(self.nodes)], dtype=float)

        # Free convection
        self.temperature = self.cooling.block_cooling(
            temperature=self.temperature,
            surface_mask=self.free_surface_mask,
            h=self.convection["free"],
            cp=cp,
            ambient_temperature=self.material_settings["ambient_temperature"],
            block_time=self.block_time
        )

        self.tracker.record_probe_state(
            temperature=self.temperature,
            time=self.time + self.block_time,
            stage="after_convection",
            block=block
        )

        self.tracker.print_temperature_status(
            temperature=self.temperature,
            time=self.time + self.block_time,
            label="After convection",
            block=block
        )

        self.time += self.block_time

        self.tracker.record_state(
            temperature=self.temperature,
            time=self.time,
            stage="after_block",
            block=block,
            laser_position=block.center
        )

        self.tracker.record_deposition_frame(
            temperature=self.temperature,
            time=self.time,
            block=block,
            stage="after_block"
        )
     

    # ----------------------------------
    def simulate_dwell(self):

        if self.dwell_dt <= 0:
            raise ValueError("Dwell timestep must be positive.")

        remaining = self.dwell_time

        while remaining > 1e-12:

            dt = min(self.dwell_dt, remaining)

            # Step 1: Conduction
            alpha = self.current_diffusivity()

            self.temperature = self.conduction.step(
                temperature=self.temperature,
                dt=self.block_time,
                active_mask=self.active_mask(),
                volumes=self.node_volumes,
                edge_areas=self.edge_areas
            )

            # Step 2: Forced convection
            cp = np.array([
                node.material.specific_heat(self.temperature[i])
                for i, node in enumerate(self.nodes)
            ], dtype=float)

            self.temperature = self.cooling.dwell_cooling(
                temperature=self.temperature,
                surface_mask=self.forced_surface_mask,
                h=self.convection["forced"],
                cp=cp,
                ambient_temperature=self.material_settings["ambient_temperature"],
                dt=dt
            )

            # Step 3: Free convection
            cp = np.array([
                node.material.specific_heat(self.temperature[i])
                for i, node in enumerate(self.nodes)
            ], dtype=float)

            self.temperature = self.cooling.dwell_cooling(
                temperature=self.temperature,
                surface_mask=self.free_surface_mask,
                h=self.convection["free"],
                cp=cp,
                ambient_temperature=self.material_settings["ambient_temperature"],
                dt=dt
            )

            # Advance simulation time
            self.time += dt
            remaining -= dt

            # Record simulation state
            self.tracker.record_state(
                temperature=self.temperature,
                time=self.time,
                stage="dwell",
                block=None,
                laser_position=None
            )

            # Record thermocouples
            self.tracker.record_probe_state(
                temperature=self.temperature,
                time=self.time,
                stage="dwell",
                block=None)
            
    # --------------------------------------
    def run(self):

        self.tracker.record_probe_state( temperature=self.temperature, time=self.time, stage="initial" )
        self.wall_start = datetime.now()
        compute_start = time.perf_counter()

        print("----------------------------")
        print("Beginning deposition")

        print(f"\nSimulation start:", self.wall_start.strftime("%Y-%m-%d %H:%M:%S"))
        print("----------------------------")

        print("\n------ PROCESS DISCRETIZATION ------")
        print(f"Deposition length : " f"{self.geometry.length * 1000:.3f} mm")
        print(f"Blocks per layer  : " f"{self.blocks_per_layer}")
        print(f"Block length: " f"{self.block_length * 1000:.3f} mm")
        print(f"Travel speed: " f"{self.waam['scan_speed'] * 1000:.3f} mm/s")
        print(f"Time per block: " f"{self.block_time:.4f} s")
    

        current_layer = None

        ordered_blocks = self.geometry.deposition_order()

        for block in ordered_blocks:

            if block.layer != current_layer:

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

            self.simulate_block(block)


        # Final cooling period
        if current_layer is not None:

            print(
                f"Dwell after final layer "
                f"{current_layer + 1}"
            )

            self.simulate_dwell()

        self.computation_time = time.perf_counter() - compute_start
        self.wall_end = datetime.now()

        print("-------------------")
        print("Simulation complete!")

        print(f"\nSimulation end:",
            self.wall_end.strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"\nSimulated process time: "
              f"{self.time:.2f} s")
        
        print(f"\nComputer calculation time: "
            f"{self.computation_time:.2f} s")

        self.tracker.finalize()

        return self.tracker.history


    # --------------------------------------
    def maximum_temperature(self):
        
        return np.max(self.temperature)
