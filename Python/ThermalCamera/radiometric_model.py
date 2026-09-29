import pandas as pd
import os
import numpy as np
import cv2 
import skimage
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


#---------------------------------
'''
Factory/radiometric camera calibration 
Constants such as R, B, F, J0, J1, X, alpha1, alpha2, Beta1, Beta2 are stored throught the camera
You are not experimentally determining these yourself 

'''
class RadiometricModel:

    def __init__(self, camera_constants, environment, emissivity):

        self.R = camera_constants["R"]
        self.B = camera_constants["B"]
        self.F = camera_constants["F"]

        self.X = camera_constants["X"]
        self.A1 = camera_constants["A1"]
        self.A2 = camera_constants["A2"]
        self.B1 = camera_constants["B1"]
        self.B2 = camera_constants["B2"]

        self.J1 = camera_constants["J1"]
        self.J0 = camera_constants["J0"]

        self.reflected_temperature = environment["reflected_temperature"]
        self.atmospheric_temperature = environment["atmospheric_temperature"]
        self.relative_humidity = environment["relative_humidity"]
        self.object_distance =  environment["object_distance"]
        self.external_optics_transmission = environment["external_optics_transmission"]
        self.external_optics_temperature = environment["external_optics_temperature"]
        self.emissivity = emissivity
        self.tau =  self.calculate_atmospheric_transmission()
        self.K2 = self.calculate_radiometric_correction()
        

    def calculate_atmospheric_transmission(self):

        temperature_c = self.atmospheric_temperature - 273.15 
        h2o = self.relative_humidity * np.exp(
            1.5587 + 0.06939 * temperature_c
            - 0.00027816 * temperature_c**2
            + 0.00000068455 * temperature_c**3)
        
        sqrt_distance = np.sqrt(self.object_distance)
        sqrt_h2o = np.sqrt(h2o)
        tau = (self.X * 
               np.exp(-sqrt_distance * (self.A1 + self.B1 * sqrt_h2o))
            + (1.0 - self.X) * np.exp(-sqrt_distance * (self.A2 + self.B2 * sqrt_h2o))
        )

        return tau
    

    def calculate_radiometric_correction(self):
        reflected = ( 
            (1.0 - self.emissivity) / self.emissivity
            * self.R /
            (np.exp(self.B / self.reflected_temperature) - self.F)
        )
        

        atmosphere = (
            (1.0 - self.tau) /
            (self.emissivity * self.tau) 
            * self.R /
            (np.exp(self.B / self.atmospheric_temperature) - self.F)
            )

        optics_transmission = self.external_optics_transmission
        
        optics = (
            (1.0 - optics_transmission) /
            (self.emissivity * self.tau * optics_transmission)
            * self.R /
            (np.exp(self.B / self.external_optics_temperature) - self.F)
        )

        return reflected + atmosphere + optics
        
       
    def raw_to_temperature(self, raw_data):

        radiance = (raw_data - self.J0) / self.J1
        denominator = (radiance /(self.emissivity * self.tau)) - self.K2
        temperature_k = self.B / np.log(self.R / denominator + self.F)
    
        return temperature_k - 273.15


    def temperature_to_raw(self, temperature_c):

        temperature_k = np.asarray(temperature_c, dtype=float) + 273.15
        signal = (self.K2 - 
                  self.R / (self.F - np.exp(self.B / temperature_k))
                  )
        
        raw_data = self.J1 * self.tau * self.emissivity * signal + self.J0

        return raw_data


#--------------------
'''
Emissivity calibration 
This is where we try to match the emissivity to sensor (FLIR or thermocouple) temperature 

'''
       
def correct_emissivity(temperature_data,
                        old_emissivity, 
                        new_emissivity,
                        camera_constants,
                        environment):

    old_model = RadiometricModel(camera_constants, environment, old_emissivity)
    new_model = RadiometricModel(camera_constants, environment, new_emissivity)
    raw_data = old_model.temperature_to_raw(temperature_data)
    corrected_temperature = new_model.raw_to_temperature(raw_data)

    return corrected_temperature


def solve_effective_emissivity(
    raw_data,
    true_temperature_c,
    camera_constants,
    environment,
    emissivity_bounds = (0.05, 1.0)):

    raw_value = np.asarray(raw_data, dtype=float)

    def objective(emissivity):

        model = RadiometricCalibration(camera_constants, environment, emissivity)
        predicted_temperature = model.raw_to_temperature(raw_value)
        error = predicted_temperature - true_temperature_c

        return np.mean(error**2)

    result = minimize_scalar(objective, bounds=emissivity_bounds, method="bounded")

    return result.x


def apply_emissivity_mask(raw_data,
    mask,
    emissivity_region_0,
    emissivity_region_1,
    camera_constants,
    environment):

    model_0 = RadiometricCalibration(camera_constants, environment, emissivity_region_0)
    model_1 = RadiometricCalibration(camera_constants, environment, emissivity_region_1)
    corrected_temperature = np.empty(raw_data.shape, dtype=float)

    corrected_temperature[mask == 0] = (model_0.raw_to_temperature(raw_data[mask == 0]))
    corrected_temperature[mask == 1] = (model_1.raw_to_temperature(raw_data[mask == 1]))

    return corrected_temperature



#------------------------------
class TemperatureDependentEmissivity:

    def __init__(
        self,
        temperatures_c,
        emissivities
    ):

        self.temperatures_c = np.asarray(
            temperatures_c,
            dtype=float
        )

        self.emissivities = np.asarray(
            emissivities,
            dtype=float
        )

        if (
            len(self.temperatures_c)
            != len(self.emissivities)
        ):
            raise ValueError(
                "Temperature and emissivity arrays "
                "must have the same length."
            )


    def get_emissivity(
        self,
        temperature_c
    ):

        emissivity = np.interp(
            temperature_c,
            self.temperatures_c,
            self.emissivities,
            left=self.emissivities[0],
            right=self.emissivities[-1]
        )

        return emissivity


def solve_emissivity_at_temperature(
    raw_samples,
    true_temperature_c,
    camera_constants,
    environment,
    emissivity_bounds=(0.05, 1.0)
):

    raw_samples = np.asarray(
        raw_samples,
        dtype=float
    )

    def objective(emissivity):

        model = RadiometricCalibration(
            camera_constants,
            environment,
            emissivity
        )

        camera_temperature = (
            model.raw_to_temperature(
                raw_samples
            )
        )

        error = (
            camera_temperature
            - true_temperature_c
        )

        return np.mean(
            error**2
        )


    result = minimize_scalar(
        objective,
        bounds=emissivity_bounds,
        method="bounded"
    )

    return result.x


def raw_to_temperature_variable_emissivity(
    raw_data,
    emissivity_model,
    camera_constants,
    environment,
    initial_emissivity=0.3,
    tolerance=0.01,
    max_iterations=20
):

    emissivity = np.full(
        np.shape(raw_data),
        initial_emissivity,
        dtype=float
    )

    temperature = None

    for _ in range(max_iterations):

        model = RadiometricCalibration(
            camera_constants,
            environment,
            emissivity
        )

        new_temperature = (
            model.raw_to_temperature(
                raw_data
            )
        )

        new_emissivity = (
            emissivity_model.get_emissivity(
                new_temperature
            )
        )

        if temperature is not None:

            error = np.nanmax(
                np.abs(
                    new_temperature
                    - temperature
                )
            )

            if error < tolerance:
                break

        temperature = new_temperature
        emissivity = new_emissivity

    return temperature

    
