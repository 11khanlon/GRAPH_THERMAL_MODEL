import matplotlib.pyplot as plt

from geometry import Geometry
from nodes import NodeGenerator
from graph import ThermalGraph
from config import (MATERIAL,GRAPH, CONVECTION, BLOCK, DWELL, BUILD, SENSORS, WAAM)
from thermocouples import ThermalProbe, middle_block_of_layer, check_probe_location

stl_file = BUILD["stl_file"]
substrate_material = Al6061Material()
wire_material = Al5183Material()
geometry = Geometry(stl_file, substrate_material=substrate_material, wire_material=wire_material)

geometry.build_blocks()

generator = NodeGenerator(geometry)

generator.generate()


#Extract information for real thermocouples 
thermocouples = []

for name, sensor in SENSORS.items():

    position = np.array([sensor["x"], sensor["y"], geometry.substrate_top + sensor["z"]])

    probe = ThermalProbe(
        name=name,
        position=position,
        region=sensor["region"]
    )

    check_probe_location(probe, geometry)
    probe.summary()
    thermocouples.append(probe)

#artificial meltpool thermocouple - used for debugging
middle_block = middle_block_of_layer(geometry, layer=0)

meltpool_probe = ThermalProbe(
    name="meltpool_probe",
    position=middle_block.center,
    region="deposition",
    layer=0)

check_probe_location(meltpool_probe, geometry)

meltpool_probe.summary()

all_probes = thermocouples + [meltpool_probe]

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
    convection=CONVECTION,
    graph_settings=GRAPH,
    dwell_info = DWELL, 
    dwell_time = selected_dwell,
    block_info = BLOCK,
        build_info=BUILD,
    probes = all_probes, 
    waam = WAAM)

history = simulation.run()

tc_names = ["TC_0", "TC_1", "TC_2", "TC_3"]
tc_temperature = np.column_stack([
    np.asarray(simulation.probe_history[name], dtype=float)
    for name in tc_names])

probe_time = np.asarray(simulation.probe_history_time, dtype=float)
meltpool_temperature = np.asarray(simulation.probe_history["meltpool_probe"], dtype=float)
maximum_tc_temp = np.nanmax(tc_temperature, axis=0)
minimum_tc_temp = np.nanmin(tc_temperature, axis=0)
final_tc_temp = tc_temperature[-1, :]



meltpool_temperature = np.asarray(simulation.probe_history["meltpool_probe"], dtype=float)
probe_time = np.asarray(simulation.probe_history_time, dtype=float)

print("\n------ SIMULATION SUMMARY ------")
print("Maximum global temperature:", simulation.maximum_temperature())
print("Stored states:", history.shape[0])
print("Final simulated time:", simulation.history_time[-1], "s")

print("\n------ THERMOCOUPLE SUMMARY ------")
for i, name in enumerate(tc_names):
    print(f"\n{name}")
    print(f"  Maximum: " f"{maximum_tc_temp[i]:.2f} °C")
    print(f"  Minimum: " f"{minimum_tc_temp[i]:.2f} °C")
    print(f"  Final: " f"{final_tc_temp[i]:.2f} °C")


# Save raw history
np.savez_compressed(
"thermal_history.npz",

# Full graph history
temperature=simulation.history,
time=simulation.history_time,
active_history=simulation.active_history,
stage_history=simulation.stage_history,
source_position_history=simulation.laser_position_history,

# Geometry
node_positions=node_positions,
substrate_mask=substrate_mask,
node_block_ids=node_block_ids,

# Thermocouples
probe_time=np.asarray(simulation.probe_history_time),
probe_stage=np.asarray(simulation.probe_history_stage),
probe_block=np.asarray(simulation.probe_history_block, dtype=object),

tc_names=np.asarray(tc_names),
tc_temperature=tc_temperature,

tc_positions=np.asarray([
    probe.position
    for probe in thermocouples
    ])
)

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
