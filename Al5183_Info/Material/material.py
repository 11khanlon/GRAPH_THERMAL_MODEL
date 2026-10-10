#Temperature-dependent material properties for LM5(5182) Aluminum alloy


import numpy as np
from config import MATERIAL

class LM5Material:

    def __init__(self):

        # Density (constant)
        self.rho = MATERIAL["density"]      # kg/m^3

        #----- k information --------
        self.k_temperature_table = np.array([25, 100, 200, 300, 400, 500, 542, 633, 633, 700, 800])
        self.k_table = np.array([85, 103, 116, 128, 133, 139, 100, 80, 63, 65, 68])

        #--------- cp information --------
        self.cp_temperature_table = np.array([25, 100, 200, 300, 400, 500, 542, 633, 633, 700, 800])
        self.cp_table = np.array([0.92, 0.98, 0.99, 1.06, 1.105, 1.19, 1.19, 1.20, 1.22, 1.22, 1.22]) * 1000
                

    #--------- Thermal conductivity ---------------

    def conductivity(self, T):
        return np.interp(T, self.k_temperature_table, self.k_table)
    
   
    # ----------- Specific heat --------------

    def specific_heat(self, T):
        return np.interp(T, self.cp_temperature_table, self.cp_table)
    

    # --------- Thermal diffusivity -------------------

    def diffusivity(self, T):

        k = self.conductivity(T)
        cp = self.specific_heat(T)
        alpha = k / (self.rho * cp)
        return alpha


    # -------- Average layer diffusivity -----------------

    def layer_diffusivity(self, temperatures):

        Tavg = np.mean(temperatures)

        return self.diffusivity_linear_si(Tavg)
    


    #--------- Print material summary ------------

    def summary(self):

        print("Material : LM5(5182) Aluminum alloy")

        print(f"Density : {self.rho} kg/m³")

        print(f"k(25°C) = {self.conductivity(25):.2f} W/m-K")

        print(f"Cp(25°C) = {self.specific_heat(25):.1f} J/kg-K")

        print(f"alpha(25°C) = {self.diffusivity(25):.3e} m²/s")


#--------------------------------------------
if __name__ == "__main__":

    material = LM5Material()

    material.summary()

    temperatures = np.linspace(25, 1600, 10)

    print()

    print("Temperature-dependent properties")

    for T in temperatures:

        print(f"{T:7.1f} °C | "
            f"k = {material.conductivity(T):6.2f} | "
            f"Cp = {material.specific_heat(T):7.1f} | "
            f"alpha = {material.diffusivity(T):.3e}") 