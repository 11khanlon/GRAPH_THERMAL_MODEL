#Temperature-dependent material properties for LM5(5182) Aluminum alloy

import numpy as np



class AluminumMaterial:

    def __init__(self, name, rho, k_temperature_table, k_table, cp_temperature_table, cp_table):
        self.name = name
        self.rho = rho

        self.k_temperature_table = np.asarray(k_temperature_table, dtype=float)
        self.k_table = np.asarray(k_table, dtype=float)
        self.cp_temperature_table = np.asarray(cp_temperature_table, dtype=float)
        self.cp_table = np.asarray(cp_table, dtype=float)

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
        

class Al5183Material:

    def __init__(self):

        super().__init__(
            name = "Al5183",
            rho = 2650,      # kg/m^3
            k_temperature_table = np.array([25, 100, 200, 300, 400, 500, 542, 633, 633, 700, 800]),
            k_table = np.array([85, 103, 116, 128, 133, 139, 100, 80, 63, 65, 68]),
            cp_temperature_table = np.array([25, 100, 200, 300, 400, 500, 542, 633, 633, 700, 800]),
            cp_table = np.array([0.92, 0.98, 0.99, 1.06, 1.105, 1.19, 1.19, 1.20, 1.22, 1.22, 1.22]) * 1000
        )  

    
class Al6061Material:

    def __init__(self):

        super().__init__(
            name = "Al6061",
            rho = 2705,      # kg/m^3
            k_temperature_table = np.array([25, 100, 200, 300, 400, 500, 600, 642, 642, 700, 800]),
            k_table = np.array([185, 195, 203, 211, 212, 225, 200, 90, 90, 91, 92]),
            cp_temperature_table = np.array([25, 100, 200, 300, 400, 500, 600, 642, 642, 700, 800]),
            cp_table = np.array([ 0.87, 0.95, 0.98, 1.02, 1.06, 1.15, 1.16, 1.16, 1.17, 1.17, 1.17]) * 1000
        )  


#--------------------------------------------
if __name__ == "__main__":

    wire = Al5183Material()
    substrate = Al6061Material()

    wire.summary()
    print()
    substrate.summary()

    material.summary()

    temperatures = np.linspace(25, 1600, 10)

    print()

    print("Temperature-dependent properties")

    for T in temperatures:

        print(f"{T:7.1f} °C | "
            f"k = {material.conductivity(T):6.2f} | "
            f"Cp = {material.specific_heat(T):7.1f} | "
            f"alpha = {material.diffusivity(T):.3e}") 