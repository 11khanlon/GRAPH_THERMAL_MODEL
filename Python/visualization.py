"""
visualization.py

Visualization tools for the graph-theory DED thermal model.

Used after simulation has completed.

Provides:

1. Sensor thermal history
2. Maximum temperature history
3. 3D nodal temperature distribution
4. Active node visualization
5. Layer temperature visualization

This file does NOT perform any thermal calculations.
"""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


class Visualizer:

    def __init__(self, geometry):

        self.geometry = geometry

        self.nodes = geometry.nodes


    # ---------------------------------------------------------
    # Plot thermocouple history
    # ---------------------------------------------------------

    def temperature_history(
        self,
        time,
        temperature,
        experimental=None,
    ):

        plt.figure(figsize=(10,5))

        plt.plot(
            time,
            temperature,
            linewidth=2,
            label="Graph Theory"
        )

        if experimental is not None:

            plt.plot(
                time,
                experimental,
                "--",
                linewidth=2,
                label="Experimental"
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title("Thermal History")

        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


    # ---------------------------------------------------------
    # Plot maximum temperature
    # ---------------------------------------------------------

    def max_temperature(self, history):

        Tmax = np.max(history, axis=1)

        plt.figure(figsize=(10,5))

        plt.plot(Tmax)

        plt.xlabel("Time Step")
        plt.ylabel("Maximum Temperature (°C)")
        plt.title("Maximum Temperature During Build")

        plt.grid(True)

        plt.tight_layout()
        plt.show()


    # ---------------------------------------------------------
    # Plot 3D temperature field
    # ---------------------------------------------------------

    def temperature_field(
        self,
        temperature,
        title="Temperature Field",
    ):

        coords = np.array(
            [node.position for node in self.nodes]
        )

        fig = plt.figure(figsize=(9,8))

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        scatter = ax.scatter(

            coords[:,0],
            coords[:,1],
            coords[:,2],

            c=temperature,

            cmap="inferno",

            s=20
        )

        fig.colorbar(
            scatter,
            label="Temperature (°C)"
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        ax.set_title(title)

        plt.tight_layout()
        plt.show()


    # ---------------------------------------------------------
    # Active nodes
    # ---------------------------------------------------------

    def active_nodes(self):

        coords = []

        for node in self.nodes:

            if node.active:

                coords.append(node.position)

        coords = np.array(coords)

        fig = plt.figure(figsize=(8,7))

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        ax.scatter(

            coords[:,0],
            coords[:,1],
            coords[:,2],

            s=10
        )

        ax.set_title("Active Nodes")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        plt.show()


    # ---------------------------------------------------------
    # Plot one deposited layer
    # ---------------------------------------------------------

    def layer_temperature(
        self,
        layer,
        temperature,
    ):

        coords = []
        temps = []

        for node, T in zip(self.nodes, temperature):

            if node.layer == layer:

                coords.append(node.position)
                temps.append(T)

        coords = np.array(coords)
        temps = np.array(temps)

        fig = plt.figure(figsize=(8,7))

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        scatter = ax.scatter(

            coords[:,0],
            coords[:,1],
            coords[:,2],

            c=temps,

            cmap="inferno",

            s=25
        )

        fig.colorbar(
            scatter,
            label="Temperature (°C)"
        )

        ax.set_title(f"Layer {layer}")

        plt.tight_layout()
        plt.show()


    # ---------------------------------------------------------
    # Animation helper
    # ---------------------------------------------------------

    def snapshot(
        self,
        history,
        step,
    ):

        self.temperature_field(

            history[step],

            title=f"Timestep {step}"
        )