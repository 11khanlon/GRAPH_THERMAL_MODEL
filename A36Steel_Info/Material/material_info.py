#Temperature-dependent material properties for A36 Steel


import numpy as np
from config import MATERIAL

class A36Material:

    def __init__(self):

        # Density (constant)
        self.rho = MATERIAL["density"]      # kg/m^3

        #----- k information --------
        self.k_temperature_table = np.array([
            19.85, 106.50, 193.20, 279.90, 366.50, 453.20,
            539.90, 626.50, 713.20, 799.90, 809.90, 853.20,
            896.50, 939.90, 983.20, 1027.00, 1070.00, 1113.00,
            1157.00, 1200.00, 1300.00, 1400.00, 1500.00
        ])


        self.k_table = np.array([
            52.54, 50.75, 48.66, 46.27, 43.59, 40.60,
            37.32, 33.73, 29.85, 25.62, 25.97, 27.33,
            28.47, 29.39, 30.06, 30.51, 30.74, 30.73, 30.49, 30.02, 
            29.00, 28.00, 27.00  #estimated extension
        ])

        #-------------- cp information -----------
        self.cp_temperature_table = np.array([
            19.85, 59.14, 98.42, 137.70, 177.00,
            216.30, 255.60, 294.90, 334.10, 373.40, 412.70,
            452.00, 491.30, 530.60, 

            # ----- Estimated extension -----
            569.90, 600.00, 700.00, 750.00, 800.00, 900.00, 
            1000.00, 1100.00, 1200.00, 1300.00, 1400.00, 1500.00
        ])


        self.cp_table = np.array([

            # ----- Granta data -----

            500.9, 513.3, 521.3, 528.1, 536.5,
            548.3, 565.0, 587.5, 615.8, 649.6, 687.8,
            728.7, 770.0, 808.9, 841.7,

            # ----- Estimated extension -----

            880.0, 1050.0, 1150.0, 1050.0, 850.0, 
            760.0, 775.0, 790.0, 810.0, 840.0,  870.0      
        ])
        

    
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

        print("Material : Ti-6Al-4V")

        print(f"Density : {self.rho} kg/m³")

        print(f"k(25°C) = {self.conductivity(25):.2f} W/m-K")

        print(f"Cp(25°C) = {self.specific_heat(25):.1f} J/kg-K")

        print(f"alpha(25°C) = {self.diffusivity(25):.3e} m²/s")


#--------------------------------------------
if __name__ == "__main__":

    material = A36Material()

    material.summary()

    temperatures = np.linspace(25, 1600, 10)

    print()

    print("Temperature-dependent properties")

    for T in temperatures:

        print(f"{T:7.1f} °C | "
            f"k = {material.conductivity(T):6.2f} | "
            f"Cp = {material.specific_heat(T):7.1f} | "
            f"alpha = {material.diffusivity(T):.3e}") 