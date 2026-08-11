import numpy as np
import matplotlib.pyplot as plt
import sys 

sys.path.append('C:\\Users\\Kayleigh\\GraphThermalModel\\Python')
from config import LASER, MATERIAL

CALIBRATION = {

    "epsilon_values": np.arange(
        1.0e-3,
        6.0e-3,
        0.25e-3
    ),

    "gain_values": np.arange(
        0.5,
        2.1,
        0.1
    ),

    "case": "A",

    "experimental_file": "case_A.csv",

    "thermocouple": "TC_A",

}


def MAPE(T_exp, T_model):

    mask = T_exp != 0

    return np.mean(
        np.abs(
            (T_exp[mask] - T_model[mask])
            / T_exp[mask]
        )
    ) * 100


best_mape = np.inf
best_epsilon = None
best_gain = None

for epsilon in epsilon_values:

    for gain in gain_values:

        simulated_temperature = run_model(
            epsilon=epsilon,
            gain=gain,
            case="A"
        )

        error = MAPE(
            experimental_temperature,
            simulated_temperature
        )

        if error < best_mape:

            best_mape = error
            best_epsilon = epsilon
            best_gain = gain