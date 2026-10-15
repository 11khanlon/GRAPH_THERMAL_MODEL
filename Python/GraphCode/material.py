#Temperature-dependent material properties for LM5(5182) Aluminum alloy

import numpy as np
from config import MATERIAL


class A36Material:

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

        # Clamp temperatures to the maximum tabulated values
        T_k = np.minimum(T, self.k_temperature_table[-1])
        T_cp = np.minimum(T, self.cp_temperature_table[-1])

        # Interpolate material properties
        k = self.conductivity(T_k)
        cp = self.specific_heat(T_cp)

        # Thermal diffusivity (m²/s)
        alpha = k / (self.rho * cp)

        return alpha


    # -------- Average layer diffusivity -----------------

    def layer_diffusivity(self, temperatures):

        Tavg = np.mean(temperatures)

        return self.diffusivity(Tavg)
    


    #--------- Print material summary ------------

    def summary(self):

        print("Material : A36Steel")
        print(f"Density : {self.rho} kg/m³")
        print(f"k(25°C) = {self.conductivity(25):.2f} W/m-K")
        print(f"Cp(25°C) = {self.specific_heat(25):.1f} J/kg-K")
        print(f"alpha(25°C) = {self.diffusivity(25):.3e} m²/s")
        

class WireMaterial(A36Material):

    def __init__(self):

        super().__init__(
            name = MATERIAL["name"],
            rho = MATERIAL["density"],      # kg/m^3,      # kg/m^3
            k_temperature_table = np.array([
                        19.85, 106.50, 193.20, 279.90, 366.50, 453.20,
                        539.90, 626.50, 713.20, 799.90, 809.90, 853.20,
                        896.50, 939.90, 983.20, 1027.00, 1070.00, 1113.00,
                        1157.00, 1200.00, 1300.00, 1400.00, 1500.00
                    ]),
            k_table = np.array([
                    52.54, 50.75, 48.66, 46.27, 43.59, 40.60,
                    37.32, 33.73, 29.85, 25.62, 25.97, 27.33,
                    28.47, 29.39, 30.06, 30.51, 30.74, 30.73, 30.49, 30.02, 
                    29.00, 28.00, 27.00  #estimated extension
                ]),
            cp_temperature_table = np.array([
                    19.85, 59.14, 98.42, 137.70, 177.00,
                    216.30, 255.60, 294.90, 334.10, 373.40, 412.70,
                    452.00, 491.30, 530.60, 

                    # ----- Estimated extension -----
                    569.90, 600.00, 700.00, 750.00, 800.00, 900.00, 
                    1000.00, 1100.00, 1200.00, 1300.00, 1400.00, 1500.00
                ]),

            cp_table = np.array([

                # ----- Granta data -----

                500.9, 513.3, 521.3, 528.1, 536.5,
                548.3, 565.0, 587.5, 615.8, 649.6, 687.8,
                728.7, 770.0, 808.9, 841.7,

                # ----- Estimated extension -----

                880.0, 1050.0, 1150.0, 1050.0, 850.0, 
                760.0, 775.0, 790.0, 810.0, 840.0,  870.0      
            ])
        )  

    

class SubstrateMaterial(A36Material):

    def __init__(self):
        super().__init__(
                    name = MATERIAL["name"],
                    rho = MATERIAL["density"],      # kg/m^3,      # kg/m^3
                    k_temperature_table = np.array([
                                19.85, 106.50, 193.20, 279.90, 366.50, 453.20,
                                539.90, 626.50, 713.20, 799.90, 809.90, 853.20,
                                896.50, 939.90, 983.20, 1027.00, 1070.00, 1113.00,
                                1157.00, 1200.00, 1300.00, 1400.00, 1500.00
                            ]),
                    k_table = np.array([
                            52.54, 50.75, 48.66, 46.27, 43.59, 40.60,
                            37.32, 33.73, 29.85, 25.62, 25.97, 27.33,
                            28.47, 29.39, 30.06, 30.51, 30.74, 30.73, 30.49, 30.02, 
                            29.00, 28.00, 27.00  #estimated extension
                        ]),
                    cp_temperature_table = np.array([
                            19.85, 59.14, 98.42, 137.70, 177.00,
                            216.30, 255.60, 294.90, 334.10, 373.40, 412.70,
                            452.00, 491.30, 530.60, 
        
                            # ----- Estimated extension -----
                            569.90, 600.00, 700.00, 750.00, 800.00, 900.00, 
                            1000.00, 1100.00, 1200.00, 1300.00, 1400.00, 1500.00
                        ]),
        
                    cp_table = np.array([
        
                        # ----- Granta data -----
        
                        500.9, 513.3, 521.3, 528.1, 536.5,
                        548.3, 565.0, 587.5, 615.8, 649.6, 687.8,
                        728.7, 770.0, 808.9, 841.7,
        
                        # ----- Estimated extension -----
        
                        880.0, 1050.0, 1150.0, 1050.0, 850.0, 
                        760.0, 775.0, 790.0, 810.0, 840.0,  870.0      
                    ])
                )  
    


#--------------------------------------------
if __name__ == "__main__":



    for material in [WireMaterial(), SubstrateMaterial()]:

        print(f"\n========== {material.name} ==========")

        T_max = max(
            np.max(material.k_temperature_table),
            np.max(material.cp_temperature_table)
        )

        temperatures = [
            25,
            T_max,
            T_max + 100,
            T_max + 500,
            T_max + 1000
        ]

        print("\nTemperature | Conductivity | Specific Heat | Diffusivity")
        print("-" * 70)

        for T in temperatures:

            k = material.conductivity(T)
            cp = material.specific_heat(T)
            alpha = material.diffusivity(T)

            print(
                f"{T:10.1f} °C | "
                f"{k:12.3f} | "
                f"{cp:13.3f} | "
                f"{alpha:.8e}"
            )

        # Test that diffusivity stays constant above maximum temperature
        assert np.isclose(
            material.diffusivity(T_max),
            material.diffusivity(T_max + 1000),
            rtol=1e-10,
            atol=0
        ), "FAILED: Diffusivity is not clamped."

        # Test layer diffusivity
        layer_temperatures = np.array([1600, 1700, 1800])

        alpha_layer = material.layer_diffusivity(layer_temperatures)

        assert np.isclose(
            alpha_layer,
            material.diffusivity(np.mean(layer_temperatures)),
            rtol=1e-10,
            atol=0
        ), "FAILED: Layer diffusivity calculation is incorrect."

        print("\nPASS: Diffusivity and layer diffusivity work correctly.")

        material.summary()
    

    