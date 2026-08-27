# Main deposition simulation

import numpy as np

from conduction import ConductionSolver
from convection import ConvectionSolver
from heat_source import GoldakHeatSource
from material import Ti64Material
from datetime import datetime
import time 
from pathlib import Path

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
                dwell_time, 
                block_info, 
                probes = None):

        self.geometry = geometry
        self.node_generator = node_generator
        self.nodes = node_generator.nodes
        self.graph = graph
        self.dwell_info = dwell_info
        self.dwell_time = dwell_time
        self.block_info = block_info


        #----- Configuration libraries ----------
        self.material_settings = material
        self.laser = laser
        self.convection = convection
        self.graph_settings = graph_settings

        #------ Temperature dependent material model ---------
        self.material_model = Ti64Material()
        self.current_alpha = (self.material_model.diffusivity_linear_si(
        self.material_settings["ambient_temperature"]
            )
        )

        #-------- Surface band definition -----------
        self.surface_thickness = block_info["surface_thickness"]

        #------- Initial temperature state ------------
        self.temperature = np.full(len(self.nodes),
                                   material["ambient_temperature"],
                                   dtype = float)

        #---------- Constants concerning time --------------
        self.block_time = block_info["time_per_block"]    #Process timing
        self.dwell_dt = dwell_info["time_step"]           # Paper uses 1-second dwell integration
        self.time = 0.0                              #simulation time 

        #--------- Initial state as history -------------
        self.history = [self.temperature.copy()]
        self.history_time = [self.time]

        #------- Solvers ---------- 
        self.conduction = ConductionSolver(graph,
            gain = graph_settings["gain"])

        self.cooling = ConvectionSolver(
            density = material["density"],
            block_length = block_info["length"])

        self.heat_source = GoldakHeatSource(
            laser_power =  laser["power"],
            conductivity = laser["thermal_conductivity"],
            alpha = laser["thermal_diffusivity"],
            scan_speed = laser["scan_speed"],
            scaling_factor = laser["goldak_C"],
            meltpool_temperature = laser["meltpool_temperature"],
            liquidus_temperature = material["liquidus_temperature"])

        #------------ Thermocouple History -----------------
        self.probes = probes if probes is not None else []

        self.probe_history = {
            probe.name: []
            for probe in self.probes}

        self.probe_history_time = []
        self.probe_history_stage = []
        self.probe_history_block = []

        #----------- Simulation metadata ------------
        self.wall_start = None
        self.wall_end = None
        self.computation_time = None
        self.active_history = [self.active_mask().copy()]
        self.active_block_history = [None]
        self.stage_history = ["initial"]
        self.laser_position_history = [
            np.array([np.nan, np.nan, np.nan])
        ]

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

    

    #-------------------------------------------
    def deposition_temperature_history(self):

        deposition_indices = np.array([
            node.id
            for node in self.nodes
            if not node.is_substrate
        ])

        return (self.history[:, deposition_indices])

    def substrate_temperature_history(self):

        substrate_indices = np.array([
            node.id
            for node in self.nodes
            if node.is_substrate])

        return (self.history[:, substrate_indices])

    #-----------------------------------------
    def block_temperature_history(self, block):

        indices = np.array([
            node.id
            for node in block.nodes
        ])

        return np.mean(self.history[:, indices], axis=1)

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

        return self.material_model.layer_diffusivity(T_layer)

    #------------------------------------------
    def print_temperature_status(self, label, block=None):

        active = self.active_mask()
        inactive = ~active

        print(f"\n--- {label} ---")

        if block is not None:
            print(
                f"Block {block.id}, "
                f"Layer {block.layer}"
            )

        print(
            f"Simulation time : "
            f"{self.time:.3f} s"
        )

        print(
            f"Global Tmin/Tmax: "
            f"{self.temperature.min():.2f} / "
            f"{self.temperature.max():.2f} °C"
        )

        print(
            f"Active nodes    : "
            f"{np.sum(active)}"
        )

        print(
            f"Active Tmax     : "
            f"{self.temperature[active].max():.2f} °C"
        )

        if np.any(inactive):

            print(
                f"Inactive Tmax   : "
                f"{self.temperature[inactive].max():.2f} °C"
            )
    
    #-----------------------------------------
    def record_state(
        self,
        stage,
        block=None,
        laser_position=None):

        # Full temperature state
        self.history.append(self.temperature.copy())

        # Simulation time
        self.history_time.append(self.time)

        # Which nodes physically exist
        self.active_history.append(
            self.active_mask().copy()
        )

        # Which deposition block is currently being processed
        if block is None:

            self.active_block_history.append(None)

        else:

            self.active_block_history.append(block.id)

        # State label
        self.stage_history.append(stage)

        # Laser position
        if laser_position is None:

            self.laser_position_history.append(
                np.array([np.nan, np.nan, np.nan])
            )

        else:

            self.laser_position_history.append(
                np.asarray(laser_position, dtype=float).copy()
            )

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

        print("Inactive nodes > ambient + 10 C:",
            np.sum(Tinactive > ambient + 10.0))

    #------------------------------------------------
    def hottest_inactive_node(self):

        active = self.active_mask()
        inactive = ~active

        if not np.any(inactive):
            return

        inactive_indices = np.where(inactive)[0]

        local_index = np.argmax(
            self.temperature[inactive]
        )

        node_index = inactive_indices[ local_index]
        node = self.nodes[node_index]

        print(
            f"Hottest inactive node: "
            f"{node.id}"
        )

        print(
            f"Temperature: "
            f"{self.temperature[node_index]:.2f} °C"
        )

        print(
            f"Layer: "
            f"{node.block.layer}"
        )

        print(
            f"Position: "
            f"{node.position}"
        )

    #----------------------------------------
    def print_current_block_temperature(
        self,
        block,
        label
    ):

        indices = np.array([
            node.id
            for node in block.nodes
        ])

        Tblock = self.temperature[
            indices
        ]

        print(
            f"{label} | "
            f"Block {block.id}: "
            f"Tmin={Tblock.min():.2f} °C, "
            f"Tmax={Tblock.max():.2f} °C"
        )

    def record_probe_state(
        self,
        stage,
        block=None
    ):

        for probe in self.probes:

            T_probe = (
                probe.interpolate_temperature(
                    nodes=self.nodes,
                    temperature=self.temperature,
                    k=8,
                    power=2,
                    active_only=True
                )
            )

            self.probe_history[
                probe.name
            ].append(
                T_probe
            )

            print(
                f"{stage} | "
                f"{probe.name} | "
                f"T={T_probe:.2f} °C"
            )

        self.probe_history_time.append(
            self.time
        )

        self.probe_history_stage.append(
            stage
        )

        if block is None:

            self.probe_history_block.append(
                None
            )

        else:

            self.probe_history_block.append(
                block.id
            )
  
    # -----------------------------------------------

    def simulate_block(self, block):

        print(
            "Temperature range:",
            self.temperature.min(),
            self.temperature.max()
        )

        print(
            "Active nodes:",
            np.sum(self.active_mask())
        )

        print(
            "Inactive nodes:",
            np.sum(~self.active_mask())
        )

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

        self.print_temperature_status("After Goldak", block)
        self.print_current_block_temperature(block, "After Goldak")

        imax = np.argmax(self.temperature)
        hot_node = self.nodes[imax]

        print(
            f"Hottest node after Goldak: "
            f"id={hot_node.id}, "
            f"block={hot_node.block_id}, "
            f"layer={hot_node.block.layer}, "
            f"T={self.temperature[imax]:.2f} °C, "
            f"position={hot_node.position}"
        )

        self.record_probe_state(stage="after_goldak", block=block)

  
        # Step 2 -> Conduction
        alpha = self.current_alpha

        self.temperature = (
            self.conduction.block_conduction(
            self.temperature,
            alpha,
            self.block_info["time_per_block"]
                )
            )
        self.record_probe_state(stage="after_conduction", block=block)
        self.print_temperature_status("After conduction", block)

        # Step 3 --> Convection
        cp = self.material_model.specific_heat(self.temperature)

        overlap = (
            self.forced_surface_mask
            & self.free_surface_mask
        )

        if np.any(overlap):
            print(
                "WARNING: forced/free overlap:",
                np.sum(overlap)
            )

        self.record_probe_state(stage="Probe after goldak", block=block)

        # Forced convection + equivalent radiation
        self.temperature = self.cooling.block_cooling(
            temperature=self.temperature,
            surface_mask=self.forced_surface_mask,
            h=self.convection["forced"],
            cp=cp,
            ambient_temperature = self.material_settings["ambient_temperature"],
            block_time = self.block_info["time_per_block"]
        )

        # Free convection
        self.temperature = self.cooling.block_cooling(
            temperature=self.temperature,
            surface_mask=self.free_surface_mask,
            h=self.convection["free"],
            cp=cp,
            ambient_temperature=self.material_settings["ambient_temperature"],
            block_time=self.block_info["time_per_block"]
        )

        self.print_temperature_status("After convection", block)

        self.time += self.block_time

        self.record_state(
        stage="after_block",
        block=block,
        laser_position=block.center)
     

    # ----------------------------------
    def simulate_dwell(self):

        steps = int(np.ceil(self.dwell_time / self.dwell_dt))

        for _ in range(steps):

            alpha = self.current_alpha

            self.temperature = (
                self.conduction.dwell_conduction(
                    temperature=self.temperature,
                    alpha=alpha,
                    dt = self.dwell_dt
                )
            )

            cp = self.material_model.specific_heat(self.temperature)

            self.temperature = (
                self.cooling.dwell_cooling(
                    temperature=self.temperature,
                    surface_mask=self.forced_surface_mask,
                    h=self.convection["forced"],
                    cp=cp,
                    ambient_temperature=self.material_settings["ambient_temperature"],
                    dt=self.dwell_dt
                )
            )

            cp = self.material_model.specific_heat(self.temperature)

            self.temperature = (
                self.cooling.dwell_cooling(
                    temperature=self.temperature,
                    surface_mask=self.free_surface_mask,
                    h=self.convection["free"],
                    cp=cp,
                    ambient_temperature=self.material_settings["ambient_temperature"],
                    dt = self.dwell_dt
                )
            )

            self.time += self.dwell_dt

            self.record_state(
                stage="dwell",
                block=None,
                laser_position=None
            )
    
    # --------------------------------------
    def run(self):

        self.record_probe_state(stage="initial")
        self.wall_start = datetime.now()
        compute_start = time.perf_counter()

        print("----------------------------")
        print("Beginning deposition")

        print(f"\nSimulation start:", self.wall_start.strftime("%Y-%m-%d %H:%M:%S"))
        print("----------------------------")
      

        current_layer = None

        ordered_blocks = self.geometry.deposition_order()

        for block in ordered_blocks:

            if block.layer != current_layer:

                # Dwell AFTER previous layer
                if current_layer is not None:

                    print(f"Dwell after layer "
                        f"{current_layer + 1}"
                    )

                    self.simulate_dwell()
                    self.current_alpha = (self.diffusivity_from_layer(current_layer))

                current_layer = block.layer

                print(f"Layer {current_layer + 1}")

            # Deposit current block
            self.simulate_block(block)

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

        self.history = np.asarray(self.history)

        self.history_time = np.asarray(self.history_time)

        self.active_history = np.asarray(
            self.active_history,
            dtype=bool)

        self.active_block_history = np.asarray(
            self.active_block_history,
            dtype=object)

        self.stage_history = np.asarray(
            self.stage_history,
            dtype=str)

        self.laser_position_history = np.asarray(
            self.laser_position_history,
            dtype=float)

        return self.history

    # --------------------------------------

    def sensor_history(self, sensor_index):

        return self.history[:, sensor_index]

    # --------------------------------------

    def maximum_temperature(self):

        return np.max(self.temperature)

    # ------------------------------

    def final_temperature(self):

        return self.temperature

# ------------------------------------

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from geometry import Geometry
    from nodes import NodeGenerator
    from graph import ThermalGraph
    from config import (MATERIAL, LASER, GRAPH, CONVECTION, BLOCK, DWELL, BUILD, SENSORS)
    from thermocouples import ThermalProbe, middle_block_of_layer, check_probe_location

    stl_file = BUILD["stl_file"]

    geometry = Geometry(stl_file)

    geometry.build_blocks()

    generator = NodeGenerator(geometry)

    generator.generate()


    #real thermocouple
    tc_a = SENSORS["TC_A"]

    tc_a_position = np.array([
         tc_a["x"],
         tc_a["y"],
        geometry.substrate_top + tc_a["z"]
    ])


    TC_A = ThermalProbe(name="TC_A", 
                        position=tc_a_position, 
                        region="substrate")
    
    check_probe_location(TC_A, geometry)
    TC_A.summary()

    #artificial meltpool thermocouple - used for debuggin 

    middle_block = middle_block_of_layer(geometry, layer=0)

    meltpool_probe = ThermalProbe(
        name="meltpool_probe",
        position=middle_block.center,
        region="deposition",
        layer=0)

    check_probe_location(meltpool_probe, geometry)

    meltpool_probe.summary()

    #Build graph

    graph = ThermalGraph(generator.nodes)
    node_positions = np.array([node.position for node in generator.nodes])
    substrate_mask = np.array([node.is_substrate for node in generator.nodes], dtype=bool)
    node_block_ids = np.array([node.block_id for node in generator.nodes], dtype=int)
    graph.build()
    graph.degree_matrix()
    graph.laplacian()
    graph.eigensystem()

    # Choose experimental case here
    selected_dwell = DWELL["dwell"]

    simulation = DepositionSimulation(
        geometry=geometry,
        node_generator=generator,
        graph=graph,
        material=MATERIAL,
        laser=LASER,
        convection=CONVECTION,
        graph_settings=GRAPH,
        dwell_info = DWELL, 
        dwell_time = selected_dwell,
        block_info = BLOCK,
        probes = [TC_A, meltpool_probe]
    )

    history = simulation.run()

    tc_a_temperature = np.asarray(
    simulation.probe_history["TC_A"],
    dtype=float)

    meltpool_temperature = np.asarray(
        simulation.probe_history["meltpool_probe"],
        dtype=float)

    probe_time = np.asarray(
        simulation.probe_history_time,
        dtype=float)

    valid_tc = np.isfinite(tc_a_temperature)

    print("TC-A maximum temperature:",
        np.max(tc_a_temperature[valid_tc]), "°C")

    print("TC-A final temperature:",
        tc_a_temperature[valid_tc][-1], "°C")

    print("\n------ SIMULATION SUMMARY ------")

    print("Maximum global temperature:", simulation.maximum_temperature())
    print("Stored states:", history.shape[0])
    print("Final simulated time:", simulation.history_time[-1], "s")
    print("TC-A maximum temperature:", np.max(tc_a_temperature), "°C")
    print("TC-A final temperature:", tc_a_temperature[-1], "°C")

    # Save raw history


    np.savez_compressed(
    "thermal_history.npz",
    temperature=simulation.history,
    time=simulation.history_time,
    active_history=simulation.active_history,
    laser_position_history=simulation.laser_position_history,
    node_positions= node_positions,
    tc_a_position=TC_A.position,
    tc_a_temperature=tc_a_temperature, 
    substrate_mask=substrate_mask, 
    node_block_ids=node_block_ids,)

    #----------------------------------
    #Make result saving directory 
    RESULT_DIR = (
        Path(__file__).resolve().parent
        / "Validation"
        / "Results"
        / "Simulation"
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result_file = (
        RESULT_DIR
        / "thermal_simulation.npz")

    