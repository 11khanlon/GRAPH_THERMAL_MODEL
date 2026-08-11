#Figure 9 Reproduction

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('C:\\Users\\Kayleigh\\GraphThermalModel\\Python')
from config import LASER, MATERIAL


#Equation 19
def temperature(x, C):

    y = 0.0
    z = 0.0

    r = np.sqrt(x**2 + y**2 + z**2)

    # Avoid singularity at x = 0
    r = np.maximum(r, 1e-12)

    delta_T = (
        C * LASER["power"]
        / (2 * np.pi * LASER["thermal_conductivity"] * r)
        * np.exp(
            -(LASER["scan_speed"]
              / (2 * LASER["thermal_diffusivity"]))
            * (x + r)
        )
    )

    T = MATERIAL["ambient_temperature"] + delta_T

    return T



# Calculate C
def calculate_C(target_temperature):

    x_calibration = 0.75e-3
    y = 0.0
    z = 0.0

    r = np.sqrt(x_calibration**2 + y**2 + z**2)

    denominator = (
        LASER["power"]
        / (2 * np.pi * LASER["thermal_conductivity"] * r)
        * np.exp(-(LASER["scan_speed"]
                / (2 * LASER["thermal_diffusivity"]))* 
                (x_calibration + r)
        )
    )

    C = target_temperature / denominator

    # Paper-compatible scaling-factor convention
    return C / 10


#------- Graph Figure 9 ----------
# x-axis
x_mm = np.linspace(-3.0, 3.0, 1000)
x_m = x_mm * 1e-3

# Meltpool cases
meltpool_temperatures = [MATERIAL["liquidus_temperature"], 1900, 2200, 2450]


plt.figure(figsize=(8, 5))

for Tmelt in meltpool_temperatures:

    C = calculate_C(Tmelt)

    T = temperature(x_m, C)

    plt.plot(x_mm, T, label=f"$T_0$ = {Tmelt} °C, C = {C:.3f}")

    # Mark calibration point
    T_cal = temperature(np.array([0.75e-3]), C)[0]

    plt.scatter([0.75], [T_cal])



# Liquidus temperature

plt.axhline(
    MATERIAL["liquidus_temperature"],
    linestyle="--",
    label="Liquidus = 1630 °C"
)


# Calibration location
plt.axvline(
    - 0.75,
    linestyle=":",
    label="Trailing edge = 0.75 mm"
)


plt.xlabel("Meltpool length, x (mm)")
plt.ylabel("Temperature (°C)")

plt.title("Effect of Scaling Factor C on Meltpool Temperature")

plt.xlim(-3, 3)
plt.ylim(0, 3000)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()