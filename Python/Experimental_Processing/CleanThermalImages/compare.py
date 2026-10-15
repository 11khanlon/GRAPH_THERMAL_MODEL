target_time = 3.0

camera_frame = validation.find_nearest_frame(
    timing,
    target_time
)

simulation_frame = validation.find_nearest_simulation_frame(
    simulation.tracker.history_time,
    target_time
)