# Main deposition simulation

import numpy as np

from conduction import ConductionSolver
from convection import ConvectionSolver
from heat_source import ArcHeatSource
from material import WireMaterial, SubstrateMaterial
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

        self.cooling = ConvectionSolver( density = rho, block_length = self.block_length)

        self.heat_source = ArcHeatSource(
            material_model=self.wire_material,
            reference_temperature=self.material_settings["ambient_temperature"],
            liquidus_temperature=self.material_settings["liquidus_temperature"],
            waam=self.waam)

        #----------- Simulation metadata ------------
        self.wall_start = None
        self.wall_end = None
        self.computation_time = None
        self.tracker = DepositionTracker(nodes=self.nodes, initial_temperature=self.temperature,
                                          ambient_temperature=self.material_settings[ "ambient_temperature" ], probes=probes)

      

    #------------- Helper Functions -----------
    #activate nodes belonging to the newly deposited block 
    def activate_block(self, block):
        ambient = self.material_settings["ambient_temperature"]
        for node in block.nodes:
            self.temperature[node.id] = ambient
            node.active = True
        block.active = True

    def reset_inactive_temperatures(self):
        mask = self.active_mask()
        self.temperature[~mask] = (self.material_settings["ambient_temperature"])

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
            if (node.active
                and not node.is_substrate
                and node.block.layer == layer)
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
    def current_diffusivity(self):   #fix nonlinear issue. also, if greater than 1500, might not read it. 

        # Average temperature of each material
        T_substrate = np.mean([
            self.temperature[i]
            for i, node in enumerate(self.nodes)
            if node.is_substrate and node.active
        ])
    
        T_wire = np.mean([
            self.temperature[i]
            for i, node in enumerate(self.nodes)
            if not node.is_substrate and node.active
        ])
    
        # Temperature-dependent diffusivity
        alpha_substrate = self.substrate_material.diffusivity(T_substrate)
        alpha_wire = self.wire_material.diffusivity(T_wire)
    
        # Equal average of both materials
        alpha_effective = (alpha_substrate + alpha_wire) / 2
    
        return alpha_effective
    #--------------------------------------------
    def active_mask(self):

        return np.array(
            [node.active for node in self.nodes],
            dtype=bool)
    
    # -----------------------------------------------
    def simulate_block(self, block):

        self.activate_block(block)
        active = self.active_mask()

        update_exposed_faces(self.geometry)                              # Update geometry-level exposed faces
        self.node_generator.update_node_exposure(self.surface_thickness)  # Convert block faces → exposed nodes
        self.node_generator.update_surface_types()                        # Classify nodes as forced/free
        self.update_surface_masks()              # Rebuild masks
        
        #establish initial temperature stage 
        self.tracker.record_thermal_stage(time=self.time, block=block, stage="initial", temperature=self.temperature, active_mask=self.active_mask())                     

        # =========== Step 1: Arc/HeatSource heating ===========
        delta_T_arc = self.heat_source.apply(nodes = self.nodes, block = block, temperature = self.temperature)
        self.tracker.print_heat_source_increment(delta_T_arc=delta_T_arc, block=block)
         
        delta_T_arc[~active] = 0.0       # Do not apply direct heat-source increments to future material.
        self.temperature += delta_T_arc
        self.tracker.record_thermal_stage(time=self.time,  block=block, stage="heat_source", temperature=self.temperature, active_mask=self.active_mask())
        self.tracker.record_probe_state(temperature=self.temperature, time=self.time, stage="after_heatsource", block=block)
   

  
        #======== Step 2 -> Conduction ==============
        alpha = self.current_diffusivity()
        self.temperature = self.conduction.block_conduction(self.temperature, alpha, block_time = self.block_time)

        # IMPORTANT: Record inactive-node heating before resetting.
        conduction_stats = self.tracker.record_temperature_statistics(temperature=self.temperature, time=self.time + self.block_time, stage="conduction_before_reset", block=block )
        if conduction_stats["heated_inactive_count"] > 0:
            print(f"WARNING: {conduction_stats['heated_inactive_count']} inactive nodes heated above ambient + 1°C. ")
             
        self.reset_inactive_temperatures()
        
        self.tracker.record_probe_state(temperature=self.temperature, time=self.time, stage="after_conduction", block=block)
      

        self.tracker.record_thermal_stage(
            time=self.time,
            block=block,
            stage="conduction",
            temperature=self.temperature,
            active_mask=self.active_mask()
        )


        # Step 3 --> Convection
        overlap = (self.forced_surface_mask & self.free_surface_mask)

        if np.any(overlap):
            print("WARNING: forced/free overlap:", np.sum(overlap))

        # Restrict convection to active material.
        forced_surface_mask = self.forced_surface_mask & active
        free_surface_mask = self.free_surface_mask & active

        cp = np.array([node.material.specific_heat(self.temperature[node.id])
                    for node in self.nodes])
    
        # Forced convection + equivalent radiation
        self.temperature = self.cooling.block_cooling(
            temperature=self.temperature,
            surface_mask= forced_surface_mask,
            h=self.convection["forced"],
            cp=cp,
            ambient_temperature = self.material_settings["ambient_temperature"],
            block_time = self.block_time
        )

        cp = np.array([
            node.material.specific_heat(self.temperature[i])
            for i, node in enumerate(self.nodes)], dtype=float)

        self.tracker.record_thermal_stage(
            time=self.time,
            block=block,
            stage="forced_convection",
            temperature=self.temperature,
            active_mask=self.active_mask()
        )

        # Free convection
        self.temperature = self.cooling.block_cooling(
            temperature=self.temperature,
            surface_mask=free_surface_mask,
            h=self.convection["free"],
            cp=cp,
            ambient_temperature=self.material_settings["ambient_temperature"],
            block_time=self.block_time)

        self.reset_inactive_temperatures()
        self.tracker.record_probe_state(temperature=self.temperature, time=self.time + self.block_time, stage="after_convection", block=block)
        self.tracker.record_thermal_stage( time=self.time + self.block_time, block=block,stage="free_convection", temperature=self.temperature, active_mask=self.active_mask())

        self.time += self.block_time

        self.tracker.record_state(temperature=self.temperature, time=self.time, stage="after_block", block=block,laser_position=block.center)
        self.tracker.record_deposition_frame(
            temperature=self.temperature,
            time=self.time,
            block=block,
            stage="after_block")

        self.tracker.record_block_completion(block=block, time=self.time)
        self.tracker.hottest_inactive_node(temperature=self.temperature, active_mask=self.active_mask(), nodes=self.nodes)
        stats = self.tracker.record_temperature_statistics(temperature=self.temperature, time=self.time, stage="after_block", block=block)
        self.tracker.print_block_summary( block=block, time=self.time, temperature=self.temperature, active_mask=active, label="Block Complete",  record=stats )
            

    # ----------------------------------
    def simulate_dwell(self):

        if self.dwell_dt <= 0:
            raise ValueError("Dwell timestep must be positive.")

        remaining = self.dwell_time

        while remaining > 1e-12:

            dt = min(self.dwell_dt, remaining)
            active = self.active_mask()

            # Step 1: Conduction
            alpha = self.current_diffusivity()

            self.temperature = self.conduction.dwell_conduction(temperature=self.temperature, alpha=alpha, dt=dt )

            if not np.all(self.active_mask()):
                self.reset_inactive_temperatures()

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

            # Store statistics for later CSV analysis.
            self.tracker.record_temperature_statistics(temperature=self.temperature, time=self.time,
                                                        stage="dwell", block=None)
            
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

                print( f"Layer {current_layer + 1}")

            self.simulate_block(block)
            print(
                f"Deposited block {block.id}, "
                f"layer={block.layer}, "
                f"time={self.time:.2f} s, "
                f"Tmax={np.max(self.temperature):.2f} °C"
            )


        # Final cooling period
        if current_layer is not None:
            print(f"Dwell after final layer {current_layer + 1} ")
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


        # -----------------------------------------
        # Output directories

        output_dir = (
            Path.home()
            / "Downloads"
            / "Thermal_Simulation"
        )

        diagnostics_dir = output_dir / "Diagnostics"
        validation_dir = output_dir / "Validation"

        diagnostics_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        validation_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # -----------------------------------------
        # Export thermal stage table

        stage_table = self.tracker.export_thermal_stage_table(
            output_path=(
                diagnostics_dir
                / "Tables"
                / "thermal_stage_nodes.csv"
            )
        )

        # -----------------------------------------
        # Export thermal stage summary

        summary = self.tracker.create_thermal_stage_summary()

        summary_path = (
            diagnostics_dir
            / "Tables"
            / "thermal_stage_summary.csv"
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        summary.to_csv(
            summary_path,
            index=False
        )

        # -----------------------------------------
        # Export numerical diagnostics

        statistics_table = self.tracker.export_temperature_statistics(
            output_path=(
                diagnostics_dir
                / "Tables"
                / "temperature_statistics.csv"
            )
        )

        # -----------------------------------------
        # Export five-point validation histories

        tracking_table = self.tracker.export_tracking_histories(
            blocks=ordered_blocks,
            block_length=self.block_length,
            block_width=self.block_info["width"],
            output_path=(
                validation_dir
                / "simulation_tracking_history.csv"
            )
        )

        print("\nSimulation diagnostics exported.")
        print(f"Thermal stage rows: {len(stage_table)}")
        print(f"Thermal summary rows: {len(summary)}")
        print(f"Statistics rows: {len(statistics_table)}")
        print(f"Validation rows: {len(tracking_table)}")


        return self.tracker.history


  