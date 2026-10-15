import numpy as np 

class DepositionTracker:
    def __init__(self, nodes, initial_temperature, probes = None):

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


    def active_mask(self):

        return np.array([node.active for node in self.nodes], dtype=bool)

    def record_state(
        self,
        temperature,
        time,
        stage,
        block=None,
        laser_position=None
    ):

        temperature = np.asarray(temperature, dtype=float)

        if temperature.shape != (len(self.nodes),):
            raise ValueError("Temperature must contain one value per node.")

        self.history.append(temperature.copy())
        self.history_time.append(float(time))
        self.active_history.append(self.active_mask().copy())

        self.active_block_history.append(
            None if block is None else block.id
        )

        self.stage_history.append(stage)

        if laser_position is None:
            position = np.array([np.nan, np.nan, np.nan])
        else:
            position = np.asarray(laser_position, dtype=float).copy()

        self.laser_position_history.append(position)


    def record_probe_state(
        self,
        temperature,
        time,
        stage,
        block=None
    ):

        for probe in self.probes:

            T_probe = probe.interpolate_temperature(
                nodes=self.nodes,
                temperature=temperature,
                k=8,
                power=2,
                active_only=True
            )

            self.probe_history[probe.name].append(T_probe)

            print(
                f"{stage} | {probe.name} | "
                f"T={T_probe:.2f} °C"
            )

        self.probe_history_time.append(time)
        self.probe_history_stage.append(stage)

        self.probe_history_block.append(
            None if block is None else block.id
        )

    def record_deposition_frame(
        self,
        temperature,
        time,
        block,
        stage="after_block"
    ):

        active = self.active_mask()

        frame = {
            "time": float(time),
            "stage": stage,
            "block_id": block.id,
            "layer": block.layer,
            "source_position": np.asarray(
                block.center, dtype=float
            ).copy(),
            "temperature": np.asarray(
                temperature, dtype=float
            ).copy(),
            "active_mask": active.copy()
        }

        self.deposition_frames.append(frame)


    def finalize(self):

        self.history = np.asarray(self.history)
        self.history_time = np.asarray(self.history_time)

        self.active_history = np.asarray(
            self.active_history, dtype=bool
        )

        self.active_block_history = np.asarray(
            self.active_block_history, dtype=object
        )

        self.stage_history = np.asarray(
            self.stage_history, dtype=str
        )

        self.laser_position_history = np.asarray(
            self.laser_position_history, dtype=float
        )

        self.probe_history_time = np.asarray(
            self.probe_history_time,
            dtype=float
        )

        self.probe_history_stage = np.asarray(
            self.probe_history_stage,
            dtype=str
        )

        self.probe_history_block = np.asarray(
            self.probe_history_block,
            dtype=object
        )

        for name in self.probe_history:
            self.probe_history[name] = np.asarray(
                self.probe_history[name],
                dtype=float
            )

        return self.history

    def print_temperature_status(
        self,
        temperature,
        time,
        label,
        block=None
    ):

        temperature = np.asarray(temperature, dtype=float)

        active = self.active_mask()
        inactive = ~active

        print(f"\n--- {label} ---")

        if block is not None:
            print(
                f"Block {block.id}, "
                f"Layer {block.layer}"
            )

        print(f"Simulation time: {time:.3f} s")

        print(
            f"Global Tmin/Tmax: "
            f"{temperature.min():.2f} / "
            f"{temperature.max():.2f} °C"
        )

        print(f"Active nodes: {np.sum(active)}")

        if np.any(active):
            print(
                f"Active Tmax: "
                f"{temperature[active].max():.2f} °C"
            )

        if np.any(inactive):
            print(
                f"Inactive Tmax: "
                f"{temperature[inactive].max():.2f} °C"
            )

    def thermocouple_matrix(self):

        names = [probe.name for probe in self.probes]

        if not names:
            return names, np.asarray(self.probe_history_time), np.empty(
                (len(self.probe_history_time), 0)
            )

        temperatures = np.column_stack([
            self.probe_history[name]
            for name in names
        ])

        return (
            names,
            np.asarray(self.probe_history_time),
            temperatures
        )

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


    #-----------------------------------------
    def create_block_tracking_points(self, block, fraction=0.25):

        start = np.asarray(block.start, dtype=float)
        end = np.asarray(block.end, dtype=float)

        center = (start + end) / 2

        dx = (end[0] - start[0]) * fraction
        dy = (end[1] - start[1]) * fraction

        z = end[2]

        return {
            "center": np.array([center[0], center[1], z]),
            "x_minus": np.array([center[0] - dx, center[1], z]),
            "x_plus": np.array([center[0] + dx, center[1], z]),
            "y_minus": np.array([center[0], center[1] - dy, z]),
            "y_plus": np.array([center[0], center[1] + dy, z])
        }

    #---------------------------------------------
    def interpolate_tracking_temperature(self, point, temperature, active_mask, k=8, power=2):

        positions = np.asarray(
            [node.position for node in self.nodes],
            dtype=float)

        deposited_mask = np.array(
            [not node.is_substrate for node in self.nodes],
            dtype=bool)

        valid_indices = np.flatnonzero(active_mask & deposited_mask )

        if len(valid_indices) == 0:
            return np.nan

        distances = np.linalg.norm( positions[valid_indices] - point, axis=1 )

        nearest = np.argsort(distances)[:k]

        d = distances[nearest]
        node_indices = valid_indices[nearest]

        if np.any(d < 1e-12):
            return float(temperature[node_indices[np.argmin(d)]])

        weights = 1.0 / d**power
        weights /= weights.sum()

        return float(np.sum(weights * temperature[node_indices]))

    #-----------------------------------------
    def tracking_point_history(self, block, time_per_block):

        points = self.create_block_tracking_points(block)

        times = np.asarray(self.history_time)
        temperatures = np.asarray(self.history)
        active_history = np.asarray(self.active_history)

        indices = np.array(
            [node.id for node in block.nodes],
            dtype=int)

        if len(indices) == 0:
            raise ValueError(f"Block {block.id} has no nodes.")

        # Find the first frame where the block is fully active
        block_active = np.all( active_history[:, indices], axis=1 )

        activation_frames = np.flatnonzero(block_active)

        if len(activation_frames) == 0:
            return None

        activation_time = times[activation_frames[0]]

        # Wait one block interval before beginning observations
        tracking_start = activation_time + time_per_block

        valid_frames = np.flatnonzero((times >= tracking_start) & block_active )

        results = {
            "block_id": block.id,
            "layer": block.layer,
            "activation_time": activation_time,
            "time": times[valid_frames],
            "cooling_time": times[valid_frames] - activation_time }

        for name, point in points.items():

            results[name] = np.array([
                self.interpolate_tracking_temperature(
                    point=point,
                    temperature=temperatures[i],
                    active_mask=active_history[i]
                )
                for i in valid_frames ])

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