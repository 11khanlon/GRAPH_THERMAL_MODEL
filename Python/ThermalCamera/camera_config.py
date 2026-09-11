A700_FALLBACK_CONSTANTS = {

    "R": 19457.3671875,
    "B": 1517.0,
    "F": 1.5,

    "atmospheric_x": 0.7319999933242798,
    "atmospheric_alpha1": 1.2390000136974777e-08,
    "atmospheric_alpha2": 1.1094999585736787e-08,
    "atmospheric_beta1": 0.0031799999997019768,
    "atmospheric_beta2": 0.0031802000012248755,

    "J1": 9.749829292297363,
    "J0": 5877
}


#Thermal camera radiometric settings 
RADIOMETRIC_ENVIRONMENT = {

    # Temperatures in Kelvin
    "reflected_temperature": 293.15,
    "atmospheric_temperature": 293.15,

    # Fraction: 0.55 = 55% RH
    "relative_humidity": 0.55,

    # Camera-to-target distance [m]
    "object_distance": 2.0,

    # External IR optics/window
    "external_optics_transmission": 1.0,
    "external_optics_temperature": 293.15
}


# Emissivity calibration
INITIAL_EMISSIVITY = 0.30
EMISSIVITY_BOUNDS = (0.05, 1.0)
EMISSIVITY_TEMPERATURE_BINS_C = [100, 150, 200, 250, 300, 350, 400]
EMISSIVITY_BIN_HALF_WIDTH_C = 10.0



# Acquisition
FRAME_TIMEOUT_MS = 1000
SAVE_RAW_FRAMES = True
SAVE_TEMPERATURE_FRAMES = True