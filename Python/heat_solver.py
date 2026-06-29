"""
heat_solver.py

Transient graph-theory thermal solver for DED.

Implements Equations (15)-(18) from

Riensche et al.
Thermal Modeling of Directed Energy Deposition
using Graph Theory (2023)
"""

import numpy as np


class HeatSolver:

    def __init__(
        self,
        propagator,
        node_blocks,
        alpha,
        gain,
        block_time,
        dwell_time,
        heat_transfer_coeff,
        density,
        cp,
        block_length,
        ambient_temperature=25.0,
    ):

        self.P = propagator

        self.node_blocks = node_blocks

        self.alpha = alpha
        self.gain = gain

        self.block_time = block_time
        self.dwell_time = dwell_time

        self.h = heat_transfer_coeff

        self.rho = density
        self.cp = cp
        self.L = block_length

        self.Ta = ambient_temperature

        self.temperature = np.ones(
            propagator.shape[0]
        ) * ambient_temperature

        self.active = np.zeros(
            propagator.shape[0],
            dtype=bool
        )

        self.history = []

    ###########################################################
    # Activate one deposited block
    ###########################################################

    def activate_block(
        self,
        block_index,
        meltpool_temperature
    ):

        indices = np.where(
            self.node_blocks == block_index
        )[0]

        self.active[indices] = True

        self.temperature[indices] = meltpool_temperature

    ###########################################################
    # Graph conduction
    ###########################################################

    def conduction_step(self):

        T = self.temperature.copy()

        T = self.P @ T

        self.temperature[self.active] = T[self.active]

    ###########################################################
    # Convection
    ###########################################################

    def convection_step(self):

        beta = (
            self.h /
            (self.rho * self.L * self.cp)
        )

        active_temp = self.temperature[self.active]

        active_temp = (
            self.Ta +
            (active_temp - self.Ta)
            * np.exp(-beta * self.block_time)
        )

        self.temperature[self.active] = active_temp

    ###########################################################
    # Save current temperatures
    ###########################################################

    def save_history(self):

        self.history.append(
            self.temperature.copy()
        )

    ###########################################################
    # Simulate deposition of one block
    ###########################################################

    def deposit_block(
        self,
        block_index,
        meltpool_temperature
    ):

        self.activate_block(
            block_index,
            meltpool_temperature
        )

        self.conduction_step()

        self.convection_step()

        self.save_history()

    ###########################################################
    # Simulate dwell time
    ###########################################################

    def dwell(self):

        if self.dwell_time <= 0:
            return

        n_steps = int(self.dwell_time)

        for _ in range(n_steps):

            self.conduction_step()

            beta = (
                self.h /
                (self.rho * self.L * self.cp)
            )

            active_temp = self.temperature[self.active]

            active_temp = (
                self.Ta +
                (active_temp - self.Ta)
                * np.exp(-beta)
            )

            self.temperature[self.active] = active_temp

            self.save_history()

    ###########################################################
    # Run complete simulation
    ###########################################################

    def run(
        self,
        n_blocks,
        meltpool_temperature
    ):

        print("Starting thermal simulation...")

        for block in range(n_blocks):

            self.deposit_block(
                block,
                meltpool_temperature
            )

        self.dwell()

        print("Simulation complete.")

        return np.array(self.history)

    ###########################################################
    # Temperature at one node
    ###########################################################

    def node_history(self, node):

        return np.array(self.history)[:, node]

    ###########################################################
    # Maximum temperature history
    ###########################################################

    def max_temperature(self):

        return np.max(
            np.array(self.history),
            axis=1
        )