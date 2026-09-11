import numpy as np

from dataclasses import dataclass
from scipy.optimize import minimize_scalar

from radiometric_model import RadiometricModel


# ============================================================
# Calibration result
# ============================================================

@dataclass
class EmissivityCalibrationResult:

    emissivity: float

    rmse_c: float

    mae_c: float

    bias_c: float

    number_of_samples: int


# ============================================================
# Error metrics
# ============================================================

def calculate_temperature_error(
    predicted_temperature_c,
    reference_temperature_c
):

    predicted = np.asarray(
        predicted_temperature_c,
        dtype=float
    )

    reference = np.asarray(
        reference_temperature_c,
        dtype=float
    )

    error = predicted - reference

    valid = (
        np.isfinite(predicted)
        & np.isfinite(reference)
    )

    if not np.any(valid):
        raise ValueError(
            "No valid temperature samples were provided."
        )

    error = error[valid]

    rmse = np.sqrt(
        np.mean(error**2)
    )

    mae = np.mean(
        np.abs(error)
    )

    bias = np.mean(error)

    return {
        "rmse_c": rmse,
        "mae_c": mae,
        "bias_c": bias,
        "number_of_samples": len(error)
    }


# ============================================================
# Constant effective emissivity calibration
# ============================================================

def solve_effective_emissivity(
    raw_data,
    reference_temperature_c,
    camera_constants,
    environment,
    emissivity_bounds=(0.05, 1.0)
):
    """
    Find one effective emissivity value that minimizes the
    temperature error between the FLIR camera and a reference
    sensor such as a thermocouple.

    raw_data:
        Raw FLIR detector values corresponding to the
        thermocouple measurements.

    reference_temperature_c:
        Thermocouple/reference temperatures in degrees C.

    Returns:
        EmissivityCalibrationResult
    """

    raw_data = np.asarray(
        raw_data,
        dtype=float
    )

    reference_temperature_c = np.asarray(
        reference_temperature_c,
        dtype=float
    )


    # Allow a scalar reference temperature to be used with
    # multiple raw samples.
    if reference_temperature_c.ndim == 0:

        reference_temperature_c = np.full(
            raw_data.shape,
            reference_temperature_c,
            dtype=float
        )


    if raw_data.shape != reference_temperature_c.shape:

        raise ValueError(
            "raw_data and reference_temperature_c "
            "must have the same shape."
        )


    valid = (
        np.isfinite(raw_data)
        & np.isfinite(reference_temperature_c)
    )

    raw_valid = raw_data[valid]

    reference_valid = (
        reference_temperature_c[valid]
    )


    if len(raw_valid) == 0:

        raise ValueError(
            "No valid camera/thermocouple samples."
        )


    def objective(emissivity):

        model = RadiometricModel(
            camera_constants=camera_constants,
            environment=environment,
            emissivity=emissivity
        )

        predicted_temperature = (
            model.raw_to_temperature(
                raw_valid
            )
        )

        error = (
            predicted_temperature
            - reference_valid
        )

        return np.mean(
            error**2
        )


    optimization = minimize_scalar(
        objective,
        bounds=emissivity_bounds,
        method="bounded"
    )


    best_emissivity = (
        optimization.x
    )


    best_model = RadiometricModel(
        camera_constants=camera_constants,
        environment=environment,
        emissivity=best_emissivity
    )


    predicted_temperature = (
        best_model.raw_to_temperature(
            raw_valid
        )
    )


    metrics = calculate_temperature_error(
        predicted_temperature,
        reference_valid
    )


    return EmissivityCalibrationResult(

        emissivity=best_emissivity,

        rmse_c=metrics["rmse_c"],

        mae_c=metrics["mae_c"],

        bias_c=metrics["bias_c"],

        number_of_samples=
            metrics["number_of_samples"]
    )


# ============================================================
# Recalculate a temperature file using a new emissivity
# ============================================================

def correct_emissivity(
    temperature_data,
    old_emissivity,
    new_emissivity,
    camera_constants,
    environment
):
    """
    Use this only when the available data are already
    temperature values rather than raw detector counts.

    The original temperature is first converted back to its
    equivalent detector signal using old_emissivity, then
    recalculated using new_emissivity.
    """

    old_model = RadiometricModel(
        camera_constants,
        environment,
        old_emissivity
    )

    new_model = RadiometricModel(
        camera_constants,
        environment,
        new_emissivity
    )


    raw_data = (
        old_model.temperature_to_raw(
            temperature_data
        )
    )


    corrected_temperature = (
        new_model.raw_to_temperature(
            raw_data
        )
    )


    return corrected_temperature


# ============================================================
# Temperature-dependent emissivity model
# ============================================================

class TemperatureDependentEmissivity:

    def __init__(
        self,
        temperatures_c,
        emissivities
    ):

        temperatures_c = np.asarray(
            temperatures_c,
            dtype=float
        )

        emissivities = np.asarray(
            emissivities,
            dtype=float
        )


        if (
            len(temperatures_c)
            != len(emissivities)
        ):

            raise ValueError(
                "Temperature and emissivity arrays "
                "must have the same length."
            )


        if len(temperatures_c) < 2:

            raise ValueError(
                "At least two calibration temperatures "
                "are required for epsilon(T)."
            )


        order = np.argsort(
            temperatures_c
        )


        self.temperatures_c = (
            temperatures_c[order]
        )

        self.emissivities = (
            emissivities[order]
        )


    def get_emissivity(
        self,
        temperature_c
    ):

        return np.interp(

            temperature_c,

            self.temperatures_c,

            self.emissivities,

            left=self.emissivities[0],

            right=self.emissivities[-1]
        )


# ============================================================
# Fit epsilon(T) from thermocouple data
# ============================================================

def solve_temperature_dependent_emissivity(
    raw_data,
    reference_temperature_c,
    camera_constants,
    environment,
    temperature_bins_c,
    bin_half_width_c=10.0,
    emissivity_bounds=(0.05, 1.0),
    minimum_samples=5
):
    """
    Determine an effective emissivity at several thermocouple
    temperature ranges.

    Example:
        temperature_bins_c = [100, 200, 300, 400]

    For the 200 C bin with a half-width of 10 C, samples with
    TC temperatures from 190 to 210 C are used.

    Returns:
        TemperatureDependentEmissivity
        calibration_results
    """

    raw_data = np.asarray(
        raw_data,
        dtype=float
    )

    reference_temperature_c = np.asarray(
        reference_temperature_c,
        dtype=float
    )


    if (
        raw_data.shape
        != reference_temperature_c.shape
    ):

        raise ValueError(
            "raw_data and reference_temperature_c "
            "must have the same shape."
        )


    calibration_temperatures = []

    calibrated_emissivities = []

    calibration_results = []


    for target_temperature in temperature_bins_c:

        lower_temperature = (
            target_temperature
            - bin_half_width_c
        )

        upper_temperature = (
            target_temperature
            + bin_half_width_c
        )


        mask = (

            np.isfinite(raw_data)

            & np.isfinite(
                reference_temperature_c
            )

            & (
                reference_temperature_c
                >= lower_temperature
            )

            & (
                reference_temperature_c
                <= upper_temperature
            )
        )


        number_of_samples = np.sum(mask)


        if number_of_samples < minimum_samples:

            print(
                f"Skipping {target_temperature:.1f} C: "
                f"only {number_of_samples} samples."
            )

            continue


        raw_samples = (
            raw_data[mask]
        )

        tc_samples = (
            reference_temperature_c[mask]
        )


        result = solve_effective_emissivity(

            raw_data=raw_samples,

            reference_temperature_c=
                tc_samples,

            camera_constants=
                camera_constants,

            environment=
                environment,

            emissivity_bounds=
                emissivity_bounds
        )


        # Use the actual mean thermocouple temperature in
        # this range rather than simply the requested bin.
        mean_temperature = np.mean(
            tc_samples
        )


        calibration_temperatures.append(
            mean_temperature
        )

        calibrated_emissivities.append(
            result.emissivity
        )


        calibration_results.append({

            "temperature_c":
                mean_temperature,

            "emissivity":
                result.emissivity,

            "rmse_c":
                result.rmse_c,

            "mae_c":
                result.mae_c,

            "bias_c":
                result.bias_c,

            "number_of_samples":
                result.number_of_samples
        })


    if len(calibration_temperatures) < 2:

        raise RuntimeError(
            "Not enough temperature ranges contained "
            "valid samples to construct epsilon(T)."
        )


    emissivity_model = (
        TemperatureDependentEmissivity(

            temperatures_c=
                calibration_temperatures,

            emissivities=
                calibrated_emissivities
        )
    )


    return (
        emissivity_model,
        calibration_results
    )


# ============================================================
# Raw data -> temperature using epsilon(T)
# ============================================================

def raw_to_temperature_variable_emissivity(
    raw_data,
    emissivity_model,
    camera_constants,
    environment,
    initial_emissivity=0.30,
    temperature_tolerance_c=0.01,
    max_iterations=20
):
    """
    Iteratively solve the coupled problem:

        raw -> T
        T -> epsilon(T)
        epsilon(T) -> corrected T

    until the calculated temperature stops changing.
    """

    raw_data = np.asarray(
        raw_data,
        dtype=float
    )


    emissivity = np.full(
        raw_data.shape,
        initial_emissivity,
        dtype=float
    )


    previous_temperature = None


    for _ in range(max_iterations):

        model = RadiometricModel(
            camera_constants=
                camera_constants,

            environment=
                environment,

            emissivity=
                emissivity
        )


        temperature = (
            model.raw_to_temperature(
                raw_data
            )
        )


        new_emissivity = (
            emissivity_model.get_emissivity(
                temperature
            )
        )


        if previous_temperature is not None:

            maximum_change = np.nanmax(
                np.abs(
                    temperature
                    - previous_temperature
                )
            )


            if (
                maximum_change
                < temperature_tolerance_c
            ):

                return temperature


        previous_temperature = (
            temperature.copy()
        )

        emissivity = (
            new_emissivity
        )


    return temperature


# ============================================================
# Spatially varying emissivity
# ============================================================

def apply_emissivity_mask(
    raw_data,
    mask,
    emissivity_region_0,
    emissivity_region_1,
    camera_constants,
    environment
):
    """
    Apply two different emissivities to two known spatial
    regions.

        mask == 0 -> emissivity_region_0
        mask == 1 -> emissivity_region_1
    """

    raw_data = np.asarray(
        raw_data
    )

    mask = np.asarray(
        mask
    )


    if raw_data.shape != mask.shape:

        raise ValueError(
            "raw_data and mask must have "
            "the same shape."
        )


    model_0 = RadiometricModel(
        camera_constants,
        environment,
        emissivity_region_0
    )

    model_1 = RadiometricModel(
        camera_constants,
        environment,
        emissivity_region_1
    )


    temperature = np.empty(
        raw_data.shape,
        dtype=float
    )


    region_0 = (
        mask == 0
    )

    region_1 = (
        mask == 1
    )


    temperature[region_0] = (
        model_0.raw_to_temperature(
            raw_data[region_0]
        )
    )


    temperature[region_1] = (
        model_1.raw_to_temperature(
            raw_data[region_1]
        )
    )


    return temperature



emissivity_model, results = (
    solve_temperature_dependent_emissivity(
        raw_data=camera_raw,
        reference_temperature_c=tc_temperature,
        camera_constants=camera_constants,
        environment=RADIOMETRIC_ENVIRONMENT,
        temperature_bins_c=[
            100,
            150,
            200,
            250,
            300,
            350,
            400
        ],
        bin_half_width_c=10
    )
)