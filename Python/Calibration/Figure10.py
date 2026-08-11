#Figure 10 Creation 

import numpy as np
import matplotlib.pyplot as plt
import sys 

sys.path.append('C:\\Users\\Kayleigh\\GraphThermalModel\\Python')
from config import LASER, MATERIAL



# Equation (19)

def temperature(z, C):

    x = 0.0
    y = 0.0

    r = np.sqrt(
        x**2 +
        y**2 +
        z**2
    )

    r = np.maximum(r, 1e-12)

    delta_T = (
        C * LASER["power"]
        / (
            2
            * np.pi
            * LASER["thermal_conductivity"]
            * r
        )
        * np.exp(
            -(
                LASER["scan_speed"]
                / (2 * LASER["thermal_diffusivity"])
            )
            * (x + r)
        )
    )

    return (
        MATERIAL["ambient_temperature"]
        + delta_T
    )


# ---------------------------------------------------------
# Calculate C
# ---------------------------------------------------------

def calculate_C(target_temperature):

    x = 0.75e-3

    r = abs(x)

    denominator = (
        LASER["power"]
        / (
            2
            * np.pi
            * LASER["thermal_conductivity"]
            * r
        )
        * np.exp(
            -(
                LASER["scan_speed"]
                / (2 * LASER["thermal_diffusivity"])
            )
            * (x + r)
        )
    )

    C = target_temperature / denominator

    return C / 10


# ---------------------------------------------------------
# Depth
# ---------------------------------------------------------

depth_mm = np.linspace(
    0.001,
    3.0,
    1000
)

depth_m = depth_mm * 1e-3


# ---------------------------------------------------------
# Meltpool cases
# ---------------------------------------------------------

meltpool_temperatures = [MATERIAL["liquidus_temperature"], 1900, 2200, 2450]


plt.figure(figsize=(8, 5))


for Tmelt in meltpool_temperatures:

    C = calculate_C(Tmelt)

    T = temperature(
        depth_m,
        C
    )

    plt.plot(
        depth_mm,
        T,
        label=f"$T_0$ = {Tmelt} °C, C = {C:.3f}"
    )


# ---------------------------------------------------------
# 20% liquidus cutoff
# ---------------------------------------------------------

liquidus = MATERIAL["liquidus_temperature"]

cutoff_temperature = (
    0.20 * liquidus
)

plt.axhline(
    cutoff_temperature,
    linestyle="--",
    label=f"20% liquidus = {cutoff_temperature:.0f} °C"
)


# ---------------------------------------------------------
# Find depth at cutoff
# ---------------------------------------------------------

for Tmelt in meltpool_temperatures:

    C = calculate_C(Tmelt)

    T = temperature(
        depth_m,
        C
    )

    # Find first point below cutoff
    indices = np.where(
        T <= cutoff_temperature
    )[0]

    if len(indices) > 0:

        index = indices[0]

        depth_at_cutoff = depth_mm[index]

        print(
            f"{Tmelt} °C: "
            f"cutoff depth ≈ "
            f"{depth_at_cutoff:.3f} mm"
        )


plt.xlabel(
    "Depth below top surface, z (mm)"
)

plt.ylabel(
    "Temperature (°C)"
)

plt.title(
    "Subsurface Temperature Beneath Meltpool"
)

plt.xlim(0, 3)
plt.ylim(0, 2500)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()