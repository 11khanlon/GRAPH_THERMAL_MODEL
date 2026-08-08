"""
Calibration of Goldak scaling factor C

Implements Section 4.3.3

Finds C such that:
T(x=0.75mm,y=0,z=0) = desired meltpool temperature


"""

import numpy as np

from material import Ti64Material
from config import LASER


class GoldakCalibrator:

    def __init__(
        self,
        laser_power,
        conductivity,
        diffusivity,
        scan_speed):

        self.P = laser_power
        self.k = conductivity
        self.alpha = diffusivity
        self.V = scan_speed

    #--------------------
    def temperature(self, x, y, z):

        r = np.sqrt(x**2 + y**2 + z**2)
        r = max(r,1e-12)


        #Equation 19 
        T = (self.C * self.P
            / (2 * np.pi * self.k * r)
            * np.exp(
                -(self.V / (2 * self.alpha))
                * (x + r)
            )
        )

        return T



    def find_C(self, target_temperature):

        #Finds C so T(0.75mm,0,0) = desired meltpool (target) temperature
        
        #change it to r if you change the y and z 
        x = 0.75e-3
        y = 0
        z = 0


        # analytical solution

        denominator = (self.P/
                       (2*np.pi*self.k*np.sqrt(x**2))
            * np.exp(-(self.V/(2*self.alpha))*(x+abs(x)))
        )

        C = target_temperature / denominator

        return C



if __name__ == "__main__":


    calibrator = GoldakCalibrator(

        laser_power = LASER["power"],

        conductivity = LASER["thermal_conductivity"],

        diffusivity = LASER["thermal_diffusivity"],

        scan_speed = LASER["scan_speed"]

    )


    meltpool_values = [1900, 2200, 2450]


    for T in meltpool_values:


        C = calibrator.find_C(T)


        print(f"Meltpool:", T, f"\nC =", C)