import numpy as np 
import pandas as pd
from pathlib import Path


class DepositionTracker:
    def __init__(self, nodes, initial_temperature, ambient_temperature, probes = None):

        self.nodes = nodes
        self.probes = probes if probes is not None else []

        # Full temperature and activation history
        self.history = [initial_temperature.copy()]
        self.history_time = [0.0]
        self.active_history = [self.active_mask()]
        self.active_block_history = [None]
        self.stage_history = ["initial"]

        self.laser_position_history = [np.array([np.nan, np.nan, np.nan])]

        # Thermocouple history
        self.probe_history = {
            probe.name: []
            for probe in self.probes
        }

        self.probe_history_time = []
        self.probe_history_stage = []
        self.probe_history_block = []
        self.deposition_frames = []

        self.block_completion_times = {}
        self.thermal_stage_history = []


        self.ambient_temperature = float(ambient_temperature)
        self.temperature_statistics = []
        self.record_temperature_statistics(
            temperature=initial_temperature,
            time=0.0,
            stage="initial",
            block=None
        )

    #-----------------------------------------
    def _temperature(self, temperature):
        """
        Convert the supplied temperatures into a NumPy array
        and verify that there is one temperature per node.

        Parameters
        ----------
        temperature : array-like
            Temperature of every node in the simulation (°C).

        Returns
        -------
        np.ndarray
            Validated one-dimensional temperature array.
        """

        if temperature is None:
            raise ValueError(
                "Temperature is None. Check the temperature "
                "being passed from deposition.py."
            )

        temperature = np.asarray(temperature, dtype=float)

        if temperature.shape != (len(self.nodes),):
            raise ValueError(
                f"Expected {len(self.nodes)} node temperatures, "
                f"received shape {temperature.shape}."
            )

        return temperature


    # --------------------------------------------------
    # HELPER: Validate active node mask
    # --------------------------------------------------
    def _mask(self, active_mask=None):
        """
        Return a Boolean array identifying active nodes.

        If no mask is supplied, use the current node.active
        flags stored in the geometry.

        True  = substrate or already-deposited material
        False = material not yet deposited
        """

        if active_mask is None:
            active_mask = self.active_mask()

        active_mask = np.asarray(active_mask, dtype=bool)

        if active_mask.shape != (len(self.nodes),):
            raise ValueError(
                f"Expected active mask shape ({len(self.nodes)},), "
                f"received {active_mask.shape}."
            )

        return active_mask

    def _block_indices(self, block):
        """
        Return the temperature-array indices for all nodes
        belonging to a specified block.

        Parameters
        ----------
        block : Block
            The deposition block being investigated.

        Returns
        -------
        np.ndarray
            Integer indices identifying the block's nodes.

        Notes
        -----
        Assumes node.id matches the node's index in
        self.nodes and the simulation temperature array.
        """

        if block is None:
            return np.array([], dtype=int)

        indices = np.array(
            [node.id for node in block.nodes],
            dtype=int
        )

        if np.any(indices < 0) or np.any(indices >= len(self.nodes)):
            raise ValueError(
                f"Block {block.id} contains invalid node indices."
            )

        return indices

    #-----------------------------------------
    def active_mask(self):

        return np.array([node.active for node in self.nodes], dtype=bool)

    #--------------------------------------------------------------
    def record_state(self, temperature, time, stage, block=None, laser_position=None):

        temperature = np.asarray(temperature, dtype=float)

        if temperature.shape != (len(self.nodes),):
            raise ValueError("Temperature must contain one value per node.")

        self.history.append(temperature.copy())
        self.history_time.append(float(time))
        self.active_history.append(self.active_mask().copy())

        self.active_block_history.append(
            None if block is None else block.id )

        self.stage_history.append(stage)

        if laser_position is None:
            position = np.array([np.nan, np.nan, np.nan])
        else:
            position = np.asarray(laser_position, dtype=float).copy()

        self.laser_position_history.append(position)

    #--------------------------------------------------------------
    def record_probe_state(self, temperature, time, stage, block=None):

        for probe in self.probes:
            T_probe = probe.interpolate_temperature(nodes=self.nodes, temperature=temperature, k=8, power=2, active_only=True )
            self.probe_history[probe.name].append(T_probe)
            print( f"{stage} | {probe.name} | T={T_probe:.2f} ° ")
               
        self.probe_history_time.append(time)
        self.probe_history_stage.append(stage)
        self.probe_history_block.append( None if block is None else block.id)


    #--------------------------------------
    #measures information after block is deposited
    def record_deposition_frame(self,temperature, time, block, stage="after_block"):
        active = self.active_mask()

        frame = {
            "time": float(time),
            "stage": stage,
            "block_id": block.id,
            "layer": block.layer,
            "source_position": np.asarray(block.center, dtype=float).copy(),
            "temperature": np.asarray(temperature, dtype=float).copy(),
            "active_mask": active.copy()
            }

        self.deposition_frames.append(frame)


    #-----------------------------------
    def finalize(self):

        self.history = np.asarray(self.history)
        self.history_time = np.asarray(self.history_time)
        self.active_history = np.asarray(self.active_history, dtype=bool )
        self.active_block_history = np.asarray( self.active_block_history, dtype=object )
        self.stage_history = np.asarray(self.stage_history, dtype=str)
        self.laser_position_history = np.asarray( self.laser_position_history, dtype=float)
        self.probe_history_time = np.asarray(self.probe_history_time, dtype=float)
        self.probe_history_stage = np.asarray(self.probe_history_stage, dtype=str)
        self.probe_history_block = np.asarray(self.probe_history_block, dtype=object)

        for name in self.probe_history:
            self.probe_history[name] = np.asarray( self.probe_history[name], dtype=float )
        return self.history


    #------------------------------------------------
    def print_block_summary(self, block, time, temperature, active_mask=None,
                            label="Block Complete", record=None):
        """One compact printout of temperatures, active/inactive nodes, and hot spots."""
        if record is None:
            record = self.record_temperature_statistics(
                temperature, time, label, block, active_mask=active_mask
            )
        print(f"\n--- {label} | Block {block.id}, Layer {block.layer} | t={time:.3f} s ---")
        print(f"Active: {record['active_node_count']} | Inactive: {record['inactive_node_count']}")
        print(f"Global Tmin/Tmax: {record['global_min_C']:.2f} / {record['global_max_C']:.2f} °C")
        print(f"Active Tmax: {record['active_max_C']:.2f} °C | Substrate Tmax: {record['substrate_max_C']:.2f} °C | Deposit Tmax: {record['active_deposit_max_C']:.2f} °C")
        print(f"Current block Tmin/Mean/Tmax: {record['current_block_min_C']:.2f} / {record['current_block_mean_C']:.2f} / {record['current_block_max_C']:.2f} °C")
        print(f"Hottest node: {record['hottest_global_node_id']} ({record['hottest_global_node_type']}), layer={record['hottest_global_layer']}, T={record['hottest_global_temperature_C']:.2f} °C")
        if record['inactive_node_count']:
            print(f"Inactive Tmax: {record['hottest_inactive_temperature_C']:.2f} °C | Heated inactive (>ambient+1°C): {record['heated_inactive_count']}")
        if record['below_ambient_node_count']:
            print(f"WARNING: {record['below_ambient_node_count']} nodes below ambient")
        return record


    #-------------------------------------------
    def hottest_inactive_node(self, temperature, active_mask=None, nodes=None):
        """Print location of the hottest inactive node (does not store statistics)."""
        temperature = self._temperature(temperature)
        active = self._mask(active_mask)
        indices = np.flatnonzero(~active)
        if not indices.size:
            print("No inactive nodes remaining.")
            return None
        index = int(indices[np.argmax(temperature[indices])])
        node = self.nodes[index]
        print(f"Hottest inactive node: {node.id} | T={temperature[index]:.2f} °C | Layer={node.block.layer if node.block else 'N/A'} | Position={node.position}")
        return node.id


    #-----------------------------------------------------------
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


    #---------------------------------------------------
    def print_heat_source_increment(self, delta_T_arc, block=None):
        """Identify where the heat-source calculation adds the most temperature."""
        delta_T_arc = self._temperature(delta_T_arc)
        if not np.all(np.isfinite(delta_T_arc)):
            raise ValueError("Heat-source increment contains NaN or infinity")
        index = int(np.argmax(delta_T_arc))
        node = self.nodes[index]
        kind = "SUBSTRATE" if node.is_substrate else "DEPOSITED MATERIAL"
        layer = None if node.block is None else node.block.layer
        print(f"Max heat-source ΔT: {delta_T_arc[index]:.2f} °C | Node {node.id} [{kind}] | Block {node.block_id} | Layer {layer} | Position {node.position}")
        if block is not None:
            idx = self._block_indices(block)
            if idx.size:
                print(f"Current block heat-source ΔT: min={delta_T_arc[idx].min():.2f}, max={delta_T_arc[idx].max():.2f} °C")
        return node.id

  
    #---------------------------------------------------------------
    def record_temperature_statistics(self, temperature, time, stage, block=None,
                                      tolerance=1.0, active_mask=None):
        """Store extrema, validity, hottest inactive node, and block metrics.

        'heated' means above ambient+tolerance, not proof of a physical heat path.
        Call BEFORE any inactive-node reset to detect artificial preheating.
        """
        temperature = self._temperature(temperature)
        active = self._mask(active_mask)
        inactive = ~active
        finite = np.isfinite(temperature)
        deposited = np.asarray([not n.is_substrate for n in self.nodes], dtype=bool)
        substrate = ~deposited
        active_deposited = active & deposited

        def safe_min(mask):
            vals = temperature[mask & finite]
            return float(vals.min()) if vals.size else np.nan

        def safe_max(mask):
            vals = temperature[mask & finite]
            return float(vals.max()) if vals.size else np.nan

        def hottest(mask):
            indices = np.flatnonzero(mask & finite)
            if not indices.size:
                return None, np.nan, None, [np.nan] * 3
            index = int(indices[np.argmax(temperature[indices])])
            node = self.nodes[index]
            return (node.id, float(temperature[index]),
                    None if node.block is None else node.block.layer,
                    np.asarray(node.position, dtype=float).tolist())

        inactive_id, inactive_T, inactive_layer, inactive_xyz = hottest(inactive)
        global_id, global_T, global_layer, global_xyz = hottest(np.ones(len(self.nodes), dtype=bool))
        block_idx = self._block_indices(block) if block is not None else np.empty(0, dtype=int)
        block_vals = temperature[block_idx] if block_idx.size else np.empty(0)
        block_vals = block_vals[np.isfinite(block_vals)]
        block_mean = float(block_vals.mean()) if block_vals.size else np.nan
        block_min = float(block_vals.min()) if block_vals.size else np.nan
        block_max = float(block_vals.max()) if block_vals.size else np.nan
        record = {
            "time_s": float(time), "stage": stage,
            "block_id": None if block is None else block.id,
            "layer": None if block is None else block.layer,
            "global_min_C": safe_min(np.ones(len(self.nodes), dtype=bool)),
            "global_max_C": safe_max(np.ones(len(self.nodes), dtype=bool)),
            "active_min_C": safe_min(active), "active_max_C": safe_max(active),
            "substrate_max_C": safe_max(substrate & active),
            "active_deposit_max_C": safe_max(active_deposited),
            "active_node_count": int(active.sum()),
            "inactive_node_count": int(inactive.sum()),
            "heated_inactive_count": int(np.sum(inactive & finite & (temperature > self.ambient_temperature + tolerance))),
            "heated_inactive_above_10C": int(np.sum(inactive & finite & (temperature > self.ambient_temperature + 10.0))),
            "hottest_inactive_node_id": inactive_id,
            "hottest_inactive_temperature_C": inactive_T,
            "hottest_inactive_layer": inactive_layer,
            "hottest_inactive_x_m": float(inactive_xyz[0]),
            "hottest_inactive_y_m": float(inactive_xyz[1]),
            "hottest_inactive_z_m": float(inactive_xyz[2]),
            "hottest_global_node_id": global_id,
            "hottest_global_temperature_C": global_T,
            "hottest_global_layer": global_layer,
            "hottest_global_x_m": float(global_xyz[0]),
            "hottest_global_y_m": float(global_xyz[1]),
            "hottest_global_z_m": float(global_xyz[2]),
            "hottest_global_node_type": (
                None if global_id is None else
                ("substrate" if self.nodes[global_id].is_substrate else "deposited")
            ),
            "all_temperatures_finite": bool(np.all(finite)),
            "nonfinite_node_count": int(np.sum(~finite)),
            "below_ambient_node_count": int(np.sum(finite & (temperature < self.ambient_temperature - 1e-6))),
            "current_block_mean_C": block_mean,
            "current_block_min_C": block_min,
            "current_block_max_C": block_max,
        }
        self.temperature_statistics.append(record)
        if not record["all_temperatures_finite"]:
            raise ValueError(f"Non-finite temperatures at t={time:.4f}s, stage={stage}")
        return record


    def export_temperature_statistics(self, output_path):

        table = pd.DataFrame(self.temperature_statistics)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)

        return table

    #---------------------------
    def thermocouple_matrix(self):

        names = [probe.name for probe in self.probes]

        if not names:
            return names, np.asarray(self.probe_history_time), np.empty(
                (len(self.probe_history_time), 0)
            )

        temperatures = np.column_stack([
            self.probe_history[name]
            for name in names])

        return (names, np.asarray(self.probe_history_time), temperatures)

     #-------------------------------------------
    def deposition_temperature_history(self):

        deposition_indices = np.array([
            node.id
            for node in self.nodes
            if not node.is_substrate
        ])

        return (self.history[:, deposition_indices])

    #-------------------------------------
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


    #---------------------------------------------
    def record_block_completion(self, block, time):

        if block.id in self.block_completion_times:
            raise ValueError(f"Block {block.id} completion was already recorded.")

        self.block_completion_times[block.id] = float(time)



    #--------- Thermal Stage Tracker ---------------
    # -----------------------------------------
    def record_thermal_stage(
        self,
        time,
        block,
        stage,
        temperature,
        active_mask
    ):
        """
        Record the temperature distribution after a
        thermal calculation stage.

        stage:
            "initial"
            "heat_source"
            "conduction"
            "convection"

        temperature:
            Array containing temperature of every node.

        active_mask:
            Boolean array identifying active nodes.
        """

        temperature = np.asarray(
            temperature,
            dtype=float
        )

        active_mask = np.asarray(
            active_mask,
            dtype=bool
        )

        if temperature.shape != (len(self.nodes),):
            raise ValueError(
                "Temperature array size does not match node count."
            )

        if active_mask.shape != temperature.shape:
            raise ValueError(
                "Active mask size does not match temperature array."
            )

        self.thermal_stage_history.append({
            "time": float(time),
            "block_id": block.id,
            "layer": block.layer,
            "stage": stage,
            "temperature": temperature.copy(),
            "active_mask": active_mask.copy()
        })


    # -----------------------------------------
    def create_thermal_stage_table(self):

        import pandas as pd

        records = []

        positions = np.asarray(
            [node.position for node in self.nodes],
            dtype=float
        )

        for snapshot in self.thermal_stage_history:

            temperature = snapshot["temperature"]
            active_mask = snapshot["active_mask"]

            for node_index in np.flatnonzero(active_mask):

                records.append({
                    "time_s": snapshot["time"],
                    "block_id": snapshot["block_id"],
                    "layer": snapshot["layer"],
                    "stage": snapshot["stage"],
                    "node_id": self.nodes[node_index].id,
                    "x_m": positions[node_index, 0],
                    "y_m": positions[node_index, 1],
                    "z_m": positions[node_index, 2],
                    "temperature_C": temperature[node_index]
                })

        return pd.DataFrame(records)

    # -----------------------------------------
    def export_thermal_stage_summary(self, output_path):

        table = self.create_thermal_stage_summary()

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        table.to_csv(output_path, index=False)

        return table
    

    def export_thermal_stage_table(self, output_path):

        # Generate the DataFrame
        table = self.create_thermal_stage_table()

        # Create output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save to CSV
        table.to_csv(
            output_path,
            index=False
        )

        return table


    # -----------------------------------------
    def create_thermal_stage_summary(self):

        import pandas as pd

        records = []

        for snapshot in self.thermal_stage_history:

            temperature = snapshot["temperature"]
            active_mask = snapshot["active_mask"]

            deposited_mask = np.asarray(
                [not node.is_substrate for node in self.nodes],
                dtype=bool
            )

            mask = active_mask & deposited_mask

            if not np.any(mask):
                continue

            values = temperature[mask]

            records.append({
                "time_s": snapshot["time"],
                "block_id": snapshot["block_id"],
                "layer": snapshot["layer"],
                "stage": snapshot["stage"],
                "mean_deposit_temperature_C": np.mean(values),
                "max_deposit_temperature_C": np.max(values),
                "min_deposit_temperature_C": np.min(values)
            })

        return pd.DataFrame(records)


    #---------------- Obtaining points for tracking -------------------
   
    def create_block_tracking_points(self, block, block_length, block_width, fraction=0.25):

        center = np.asarray(block.center, dtype=float)

        dx = block_length * fraction
        dy = block_width * fraction

        x, y, z = center

        return {
            "center": np.array([x, y, z]),
            "x_minus": np.array([x - dx, y, z]),
            "x_plus": np.array([x + dx, y, z]),
            "y_minus": np.array([x, y - dy, z]),
            "y_plus": np.array([x, y + dy, z])
        }


    #---------------------------------------------
    def interpolate_tracking_temperature(
        self,
        point,
        temperature,
        active_mask,
        block,
        k=8,
        power=2
    ):

        indices = np.array(
            [node.id for node in block.nodes],
            dtype=int
        )

        valid_indices = indices[active_mask[indices]]

        if valid_indices.size == 0:
            return np.nan

        positions = np.asarray(
            [self.nodes[i].position for i in valid_indices],
            dtype=float
        )

        point = np.asarray(point, dtype=float)

        distances = np.linalg.norm(
            positions - point,
            axis=1
        )

        nearest = np.argsort(distances)[:k]

        d = distances[nearest]
        node_indices = valid_indices[nearest]

        if np.any(d < 1e-12):
            return float(
                temperature[node_indices[np.argmin(d)]]
            )

        weights = 1.0 / np.maximum(d, 1e-12)**power
        weights /= weights.sum()

        return float(
            np.dot(weights, temperature[node_indices])
    )
    


    def tracking_point_history( self, block, block_length, block_width):

        points = self.create_block_tracking_points(
            block=block,
            block_length=block_length,
            block_width=block_width
        )

        times = np.asarray(self.history_time, dtype=float)
        temperatures = np.asarray(self.history, dtype=float)
        active_history = np.asarray(self.active_history, dtype=bool)
        stages = np.asarray(self.stage_history)

        indices = np.array(
            [node.id for node in block.nodes],
            dtype=int
        )

        if indices.size == 0:
            raise ValueError(f"Block {block.id} has no nodes.")

        if block.id not in self.block_completion_times:
            raise ValueError(
                f"Missing completion time for block {block.id}"
            )

        simulation_times = np.asarray(
            self.history_time,
            dtype=float
        )

        deposition_end_time = self.block_completion_times[block.id]

        # Only use completed solver states.
        valid_stage = np.isin(
            stages,
            ["after_block", "dwell"]
        )

        block_active = np.all(
            active_history[:, indices],
            axis=1
        )

        valid_frames = np.flatnonzero(
            block_active
            & valid_stage
            & (times >= deposition_end_time - 1e-9)
        )

        results = {
            "block_id": block.id,
            "layer": block.layer,
            "deposition_end_time": deposition_end_time,
            "time": times[valid_frames],
            "cooling_time": (
                times[valid_frames] - deposition_end_time
            )
        }

        for name, point in points.items():

            results[name] = np.array([
                self.interpolate_tracking_temperature(
                    point=point,
                    temperature=temperatures[i],
                    active_mask=active_history[i], block = block
                )
                for i in valid_frames
            ])

        return results


    def find_nearest_simulation_frame(self, simulation_times, target_time):

        simulation_times = np.asarray(simulation_times, dtype=float)

        index = np.argmin(
            np.abs(simulation_times - target_time)
        )

        return {
            "frame_index": int(index),
            "time": float(simulation_times[index]),
            "time_error": float(
                simulation_times[index] - target_time
            )
        }


    def create_validation_schedule(self):

        records = []

        for validation_step in range(1, self.blocks_per_layer + 1):

            target_time = validation_step * self.time_per_block

            tracked_blocks = self.get_tracked_blocks(
                validation_step
            )

            for tracked in tracked_blocks:

                records.append({
                    "validation_step": validation_step,
                    "target_time_s": target_time,
                    "current_block": validation_step,
                    "tracked_block": tracked["block"],
                    "cooling_time_s": tracked["cooling_time"]
                })

        return pd.DataFrame(records)


    def export_tracking_histories(
        self,
        blocks,
        block_length,
        block_width,
        output_path
    ):

        records = []

        point_names = [
            "center",
            "x_minus",
            "x_plus",
            "y_minus",
            "y_plus"
        ]

        for block_number, block in enumerate(blocks, start=1):

            history = self.tracking_point_history(
                block=block,
                block_length=block_length,
                block_width=block_width
            )

            for i, simulation_time in enumerate(history["time"]):

                row = {
                    "block": block_number,
                    "block_id": history["block_id"],
                    "layer": history["layer"],
                    "time_s": float(simulation_time),
                    "deposition_end_time_s": float(
                        history["deposition_end_time"]
                    ),
                    "cooling_time_s": float(
                        history["cooling_time"][i]
                    )
                }

                for name in point_names:
                    row[f"{name}_T_sim_C"] = float(
                        history[name][i]
                    )

                records.append(row)

        table = pd.DataFrame(records)

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        table.to_csv(output_path, index=False)

        return table




    