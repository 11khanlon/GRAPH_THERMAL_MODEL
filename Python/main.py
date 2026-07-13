"""
Main Graph-Theory DED Thermal Simulation

Implements:
Step 1 - Build geometry
Step 2 - Graph creation
Step 3 - Layer-by-layer simulation
Step 4 - Store complete thermal history
"""

import numpy as np
import matplotlib.pyplot as plt

from config import *
from geometry import build_geometry
from graph_builder import build_graph
from goldak import compute_goldak_temperature
from conduction import conduction_step
from convection import convection_step
from dwell import dwell_step
from sensor import Sensor
from material import thermal_diffusivity

# -------------------------------------------------------
# Build geometry
# -------------------------------------------------------

print("Generating geometry...")

nodes, blocks = build_geometry()

print(f"Total Nodes : {len(nodes)}")
print(f"Total Blocks: {len(blocks)}")

# -------------------------------------------------------
# Build graph
# -------------------------------------------------------

print("Building graph...")

L, phi, K = build_graph(nodes)

print("Graph complete.")

# -------------------------------------------------------
# Sensor location
# -------------------------------------------------------

sensor = Sensor(nodes)

temperature_history = []
time_history = []

current_time = 0.0

# -------------------------------------------------------
# Initialize temperatures
# -------------------------------------------------------

T = np.ones(len(nodes)) * AMBIENT_TEMP

# -------------------------------------------------------
# Layer simulation
# -------------------------------------------------------

for layer in range(NUM_LAYERS):

    print(f"Layer {layer+1}/{NUM_LAYERS}")

    layer_blocks = [b for b in blocks if b.layer == layer]

    for block in layer_blocks:

        # --------------------------------------------
        # Activate deposited nodes
        # --------------------------------------------

        active = block.node_ids

        # --------------------------------------------
        # Goldak heat input
        # --------------------------------------------

        T = compute_goldak_temperature(
            T,
            nodes,
            block,
            MELTPOOL_TEMP
        )

        # --------------------------------------------
        # Conduction
        # --------------------------------------------

        alpha = thermal_diffusivity(np.mean(T))

        T = conduction_step(
            T,
            phi,
            K,
            alpha,
            BLOCK_TIME
        )

        # --------------------------------------------
        # Convection
        # --------------------------------------------

        T = convection_step(
            T,
            nodes
        )

        # --------------------------------------------
        # Save sensor temperature
        # --------------------------------------------

        temperature_history.append(sensor.measure(T))
        time_history.append(current_time)

        current_time += BLOCK_TIME

    # ------------------------------------------------
    # Dwell after layer
    # ------------------------------------------------

    dwell = DWELL_TIME

    for i in range(int(dwell)):

        alpha = thermal_diffusivity(np.mean(T))

        T = dwell_step(
            T,
            phi,
            K,
            alpha
        )

        temperature_history.append(sensor.measure(T))
        time_history.append(current_time)

        current_time += 1.0

# -------------------------------------------------------
# Plot result
# -------------------------------------------------------

plt.figure(figsize=(10,5))

plt.plot(
    time_history,
    temperature_history,
    linewidth=2
)

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Graph Theory Thermal History")

plt.grid(True)

plt.show()