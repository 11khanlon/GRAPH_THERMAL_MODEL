#Thermal camera radiometric settings 
RADIOMETRY = {

    # Environmental conditions
    "reflected_temperature": 293.15,   # K
    "atmospheric_temperature": 293.15, # K
    "relative_humidity": 0.55,         # 0-1

    # Camera-to-target geometry
    "object_distance": 2.0,            # m

    # External optics / IR window
    "external_optics_transmission": 1.0,
    "external_optics_temperature": 273.15,  # K

}


A700_FACTORY_CALIBRATION = {

    "R": 19457.3671875,
    "B": 1517.0,
    "F": 1.5,

    "atmospheric_x": 0.7319999933242798,
    "atmospheric_alpha1": 1.2390000136974777e-08,
    "atmospheric_alpha2": 1.1094999585736787e-08,
    "atmospheric_beta1": 0.0031799999997019768,
    "atmospheric_beta2": 0.0031802000012248755,

    "J1": 9.749829292297363,
    "J0": 5877,
}


'''
       calib_vals = [  19457.3671875, # ------------ R
                        1517.0, # ------------------- B
                        1.5, # ---------------------- F
                        0.7319999933242798, # ------- X
                        1.2390000136974777e-08, # --- A1
                        1.1094999585736787e-08, # --- A2
                        0.0031799999997019768, # ---- B1
                        0.0031802000012248755, # ---- B2
                        9.749829292297363, # -------- J1
                        5877, # --------------------- J0
                        293.15, # ------------------- TRefl
                        293.15, # ------------------- TAtm
                        20.0, # --------------------- TAtmC
                        0.55, # --------------------- Humidity
                        2, # ------------------------ Dist
                        1, # ------------------------ EOR
                        273.15, # ------------------- EOT
                    ]




'''