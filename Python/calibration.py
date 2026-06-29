"""
calibration.py

Grid-search calibration of graph parameters
(neighborhood radius ε and gain factor g)

Implements Section 4.3.4.1 from

Riensche et al.
Thermal Modeling of Directed Energy Deposition
using Graph Theory (2023)
"""

import numpy as np


class GraphCalibrator:

    def __init__(
        self,
        simulator,
        experimental_temperature
    ):
        """
        Parameters
        ----------
        simulator : callable

            Function that accepts

                epsilon
                gain

            and returns

                predicted temperature history

        experimental_temperature : ndarray
        """

        self.simulator = simulator
        self.experimental = experimental_temperature

        self.results = []

    ############################################################
    # Mean Absolute Percentage Error
    ############################################################

    @staticmethod
    def mape(measured, predicted):

        measured = np.asarray(measured)
        predicted = np.asarray(predicted)

        return np.mean(
            np.abs(
                (measured - predicted)
                / measured
            )
        ) * 100

    ############################################################
    # Root Mean Square Error
    ############################################################

    @staticmethod
    def rmse(measured, predicted):

        measured = np.asarray(measured)
        predicted = np.asarray(predicted)

        return np.sqrt(
            np.mean(
                (measured - predicted) ** 2
            )
        )

    ############################################################
    # Grid search
    ############################################################

    def grid_search(
        self,
        epsilon_values,
        gain_values
    ):

        best_error = np.inf

        best_parameters = None

        print("Beginning calibration...\n")

        for epsilon in epsilon_values:

            for gain in gain_values:

                prediction = self.simulator(
                    epsilon,
                    gain
                )

                mape = self.mape(
                    self.experimental,
                    prediction
                )

                rmse = self.rmse(
                    self.experimental,
                    prediction
                )

                self.results.append({

                    "epsilon": epsilon,

                    "gain": gain,

                    "MAPE": mape,

                    "RMSE": rmse

                })

                print(
                    f"ε={epsilon:.5f}  "
                    f"g={gain:.3f}  "
                    f"MAPE={mape:.2f}%  "
                    f"RMSE={rmse:.2f}"
                )

                if mape < best_error:

                    best_error = mape

                    best_parameters = (

                        epsilon,
                        gain,
                        rmse

                    )

        print("\nCalibration complete.")

        return best_parameters

    ############################################################
    # Results sorted by MAPE
    ############################################################

    def ranked_results(self):

        return sorted(
            self.results,
            key=lambda x: x["MAPE"]
        )

    ############################################################
    # Print best result
    ############################################################

    def summary(self):

        ranked = self.ranked_results()

        best = ranked[0]

        print()

        print("Best Calibration")

        print("----------------")

        print(
            f"Neighborhood ε : "
            f"{best['epsilon']:.5f}"
        )

        print(
            f"Gain factor g : "
            f"{best['gain']:.3f}"
        )

        print(
            f"MAPE : "
            f"{best['MAPE']:.2f}%"
        )

        print(
            f"RMSE : "
            f"{best['RMSE']:.2f} °C"
        )


if __name__ == "__main__":

    ###########################################################
    # Example
    ###########################################################

    experimental = np.random.normal(
        200,
        5,
        500
    )

    def dummy_simulator(
        epsilon,
        gain
    ):

        noise = np.random.normal(
            0,
            10,
            len(experimental)
        )

        return (
            experimental
            + noise
            + gain * 2
        )

    calibrator = GraphCalibrator(
        dummy_simulator,
        experimental
    )

    eps = np.linspace(
        0.002,
        0.004,
        5
    )

    gains = np.linspace(
        0.5,
        2.0,
        6
    )

    calibrator.grid_search(
        eps,
        gains
    )

    calibrator.summary()