"""
Synchronize experimental WAAM thermal data

Master clock: Fronius.csv -> PROCESSACTIVE

For Pass 1 and Pass 2:
    - Find deposition start: False -> True
    - Find deposition end:   True -> False
    - Keep 5 s before deposition
    - Keep 30 s dwell after deposition
    - Keep 5 s after dwell
    - Synchronize thermocouples to Fronius start
    - Insert interpolated TC value at t = 0
    - Determine corresponding A700 camera Count range

Raw files are NOT modified - Processed copies are saved separately
"""

from pathlib import Path
import numpy as np
import pandas as pd


# Paths
BASE_DIR = Path(__file__).resolve().parent

THERMAL_DATA_DIR = BASE_DIR.parent / "ThermalData"
FRONIUS_FILE = THERMAL_DATA_DIR / "Fronius.csv"
TC_FILE = THERMAL_DATA_DIR / "Thermocouples.csv"
A700_FILE = THERMAL_DATA_DIR / "FroniusA700.csv"

OUTPUT_DIR = THERMAL_DATA_DIR / "Processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Processing settings
PRE_BUFFER_S = 1
DWELL_S = 30.0
POST_DWELL_BUFFER_S = 5.0
NUMBER_OF_PASSES = 2
SCAN_SPEED_M_S = 0.762 / 60       # m/s
BEAD_LENGTH_M = 0.128             # m
x_start = 12.4/1000   #m

# Load data
fronius = pd.read_csv(FRONIUS_FILE)
tc = pd.read_csv(TC_FILE)
a700 = pd.read_csv(A700_FILE)

#-------- Clean up timestamp ---------------
def clean_timestamp_column(series):

    cleaned = (series
        .astype(str)
        .str.strip()
        .str.replace(
            r"^Timestamp\(['\"]?",
            "",
            regex=True)
        .str.replace(
            r"['\"]?\)$",
            "",
            regex=True)
    )

    return pd.to_datetime(cleaned, format="mixed", errors="raise")


# Clean timestamps
fronius["Timestamp"] = clean_timestamp_column(fronius["Timestamp"])
tc["Timestamp"] = clean_timestamp_column(tc["Timestamp"])
a700["Timestamp"] = clean_timestamp_column(a700["Timestamp"])


# Sort everything chronologically just in case
fronius = fronius.sort_values("Timestamp").reset_index(drop=True)
tc = tc.sort_values("Timestamp").reset_index(drop=True)
a700 = a700.sort_values("Timestamp").reset_index(drop=True)


# Clean Fronius PROCESSACTIVE

fronius["PROCESSACTIVE"] = (fronius["PROCESSACTIVE"]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .map({"true": True, "false": False}))


if fronius["PROCESSACTIVE"].isna().any():

    bad_values = fronius.loc[fronius["PROCESSACTIVE"].isna()]

    raise ValueError(f"PROCESSACTIVE contains values other than True/False.\n {bad_values}")


# Clean thermocouple temperatures
TC_CHANNELS = ["ch0", "ch1", "ch2", "ch3"]

for channel in TC_CHANNELS:

    
    # Example: "0030.05409" becomes: 30.05409 C
    # Remove whitespace and quotation marks first.
    tc[channel] = (
        tc[channel]
        .astype(str)
        .str.strip()
        .str.strip("'")
        .str.strip('"')
        .str.strip()
    )

    # Then convert: '0030.05409' -> 30.05409
    tc[channel] = pd.to_numeric(tc[channel], errors="coerce")


# Find Fronius deposition starts and ends
active = fronius["PROCESSACTIVE"]

# False -> True
start_mask = active & ~active.shift(fill_value=False)

# True -> False
end_mask = ~active & active.shift(fill_value=False)
start_times = fronius.loc[start_mask, "Timestamp"].reset_index(drop=True)
end_times = fronius.loc[end_mask, "Timestamp"].reset_index(drop=True)


print("\n========== FRONIUS PASSES ==========")

print(f"Detected deposition starts:{len(start_times)} ")
print(f"Detected deposition ends: {len(end_times)} ")

if len(start_times) < NUMBER_OF_PASSES:
    raise ValueError("Fewer than two deposition starts were detected") 

if len(end_times) < NUMBER_OF_PASSES:
    raise ValueError("Fewer than two deposition ends were detected")
      

# ============================================================
# Helper: interpolate thermocouples exactly at torch start

def interpolate_tc_at_time(tc_data, target_time):

    before = tc_data[tc_data["Timestamp"] <= target_time].tail(1)
    after = tc_data[tc_data["Timestamp"] >= target_time].head(1)


    if before.empty or after.empty:

        raise ValueError("Cannot interpolate thermocouple "
            f"data at {target_time}. Need data on both sides.")


    t1 = before["Timestamp"].iloc[0]
    t2 = after["Timestamp"].iloc[0]

    # Exact measurement already exists
    if t1 == target_time:

        row = before.copy()

        row["RelativeTime_s"] = 0.0
        row["InterpolatedStart"] = False

        return row


    if t2 == target_time:

        row = after.copy()

        row["RelativeTime_s"] = 0.0
        row["InterpolatedStart"] = False

        return row


    total_dt = (t2 - t1).total_seconds()
    target_dt = (target_time - t1).total_seconds()
    fraction = target_dt / total_dt

    new_row = {"Timestamp": target_time, "InterpolatedStart": True}

    for channel in TC_CHANNELS:

        T1 = before[channel].iloc[0]
        T2 = after[channel].iloc[0]

        new_row[channel] = T1 + fraction * (T2 - T1)

    return pd.DataFrame([new_row])
    


'''
# Helper: nearest A700 frame
Return the real camera frame whose timestamp is nearest to target_time
Camera frames are not interpolated because count refers to an actual JPEG
'''

def nearest_camera_frame(camera_data, target_time):

    time_difference = (camera_data["Timestamp"] - target_time).abs()

    index = time_difference.idxmin()

    return camera_data.loc[index]


def assign_stage(timestamps, deposition_start, deposition_end, dwell_end):

        conditions = [timestamps < deposition_start,
            (
                (timestamps >= deposition_start)
                & (timestamps <= deposition_end)
            ),

            (
                (timestamps > deposition_end)
                & (timestamps <= dwell_end)
            ),

            timestamps > dwell_end]

        choices = ["Pre-Deposition", "Deposition", "Dwell", "Post-Dwell"]

        return np.select(conditions, choices, default="Unknown")


def add_pass_coordinates(data, deposition_start, deposition_end, pass_number):

    data = data.copy()

    # Relative experimental time
    data["RelativeTime_s"] = (data["Timestamp"] - deposition_start).dt.total_seconds()

    # Is the torch actively depositing?
    data["IsDepositing"] = (
        (data["Timestamp"] >= deposition_start)
        & (data["Timestamp"] <= deposition_end)
    )

    # Position only exists while depositing
    data["Position_m"] = np.nan

    depositing = data["IsDepositing"]

    t = data.loc[depositing, "RelativeTime_s"]

    if pass_number == 1:

        # Pass 1: 0 m --> 0.128 m
        position = (SCAN_SPEED_M_S * t) + x_start

        data.loc[ depositing,  "Position_m"] = np.clip(position, 0.0, BEAD_LENGTH_M)

    elif pass_number == 2:

        # Pass 2: 0.128 m --> 0 m
        position = BEAD_LENGTH_M - SCAN_SPEED_M_S * t

        data.loc[depositing, "Position_m"] = np.clip(position, 0.0, BEAD_LENGTH_M)

    else:
        raise ValueError(f"Unsupported pass number: {pass_number} ")

    data["Pass"] = pass_number

    return data


# Process Pass 1 and Pass 2

for pass_index in range(NUMBER_OF_PASSES):

    pass_number = pass_index + 1
    deposition_start = start_times.iloc[pass_index]
    deposition_end =  end_times.iloc[pass_index]
    deposition_duration = (deposition_end - deposition_start).total_seconds()
    dwell_end = (deposition_end + pd.Timedelta(seconds=DWELL_S))
    window_start = ( deposition_start - pd.Timedelta(seconds=PRE_BUFFER_S))
    window_end = (dwell_end + pd.Timedelta(seconds=POST_DWELL_BUFFER_S))

  
    # Output directory for this pass
    pass_dir =  OUTPUT_DIR / f"Pass_{pass_number}"
    pass_dir.mkdir(parents=True, exist_ok=True)


    # --------- Fronius crop------------- 
    fronius_pass = fronius[(fronius["Timestamp"] >= window_start)
                           & (fronius["Timestamp"] <= window_end)].copy()

    fronius_pass = add_pass_coordinates(fronius_pass, deposition_start, deposition_end, pass_number)

    fronius_pass["Stage"] = assign_stage(fronius_pass["Timestamp"], deposition_start, deposition_end, dwell_end)


    #--------- Thermocouple crop ----------
    tc_pass = tc[(tc["Timestamp"] >= window_start)
                 & (tc["Timestamp"] <= window_end)].copy()

    print(f"\nTC full data range: {tc['Timestamp'].min()} -> {tc['Timestamp'].max()}")
    print(f"Pass {pass_number} requested TC range: {window_start} -> {window_end}")
    print(f"TC rows found in Pass {pass_number}: {len(tc_pass)} ")
    
    # Mark all original measurements
    tc_pass["InterpolatedStart"] = False

    # Interpolate an exact measurement at the Fronius deposition start
    tc_start_row = interpolate_tc_at_time(tc, deposition_start)

    # Add it without deleting any real measurements
    tc_pass = pd.concat([tc_pass, tc_start_row], ignore_index=True)
    tc_pass = (tc_pass.sort_values("Timestamp").reset_index(drop=True))

    tc_pass = add_pass_coordinates(tc_pass, deposition_start, deposition_end, pass_number)
    tc_pass["Stage"] = assign_stage(tc_pass["Timestamp"], deposition_start, deposition_end, dwell_end)

    # --------- A700 camera crop ----------------
    a700_pass = a700[(a700["Timestamp"] >= window_start)
        & (a700["Timestamp"] <= window_end)].copy()

    a700_pass = add_pass_coordinates(a700_pass, deposition_start, deposition_end, pass_number)
    a700_pass["Stage"] = assign_stage(a700_pass["Timestamp"], deposition_start, deposition_end, dwell_end)

    
    # Find important REAL camera frames
    camera_start = nearest_camera_frame(a700, deposition_start)
    camera_start_offset_s = (camera_start["Timestamp"] - deposition_start).total_seconds()
    camera_deposition_end = nearest_camera_frame(a700, deposition_end)
    camera_dwell_end = nearest_camera_frame(a700, dwell_end)
    camera_window_start = nearest_camera_frame( a700, window_start)
    camera_window_end = nearest_camera_frame(a700, window_end)


    #-------------- save processed data to csv ---------
    def save_processed_csv(data, filepath):

        output = data.copy()

        # Absolute experimental timestamp
        output["Timestamp"] = (output["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f"))

        # Explicit numeric seconds
        if "RelativeTime_s" in output.columns:
            output["RelativeTime_s"] = (
                output["RelativeTime_s"]
                .round(6)
            )

        # Position in mm
        if "TorchX_mm" in output.columns:
            output["TorchX_mm"] = (
                output["TorchX_mm"]
                .round(6)
            )

        output.to_csv(filepath, index=False )

    def organize_columns(data):

        preferred_columns = ["Timestamp", "RelativeTime_s", "Position_m", "Pass", "IsDepositing", "Stage"]

        existing_preferred = [
            column
            for column in preferred_columns
            if column in data.columns
        ]

        remaining_columns = [
            column
            for column in data.columns
            if column not in existing_preferred
        ]

        return data[
            existing_preferred
            + remaining_columns
        ]

    fronius_pass = organize_columns(fronius_pass)
    tc_pass = organize_columns(tc_pass)
    a700_pass = organize_columns(a700_pass)

    # Save processed copies
    save_processed_csv(fronius_pass, pass_dir / f"Fronius_Pass{pass_number}.csv")
    save_processed_csv(tc_pass, pass_dir / f"Thermocouples_Pass{pass_number}.csv")
    save_processed_csv(a700_pass, pass_dir / f"A700_Pass{pass_number}.csv")

    
    # Print synchronization summary
    print(f"\n========== PASS {pass_number} ==========")
    print(f"Deposition start : {deposition_start} ")
    print( f"Deposition end: {deposition_end} ")
    print(f"Deposition time: {deposition_duration:.3f} s ")
    print(f"Dwell end: {dwell_end} ")
    print(f"Window start: {window_start} ")
    print(f"Window end: {window_end} ")

    print("\n--- A700 synchronization ---")
    print("Deposition-start frame:")
    print(f"Count =  {int(camera_start['Count'])}")
    print(f"Timestamp = {camera_start['Timestamp']} ")
    print(f"Time offset =  {camera_start_offset_s:+.6f} s")

    print("\nDeposition-end frame:")
    print(f"Count = {int(camera_deposition_end['Count'])}")
    print(f"  Timestamp = {camera_deposition_end['Timestamp']} ")

    print("\nDwell-end frame:")
    print( f"Count  = {int(camera_dwell_end['Count'])} ")
    print(f"  Timestamp =  {camera_dwell_end['Timestamp']}")

    print("\nRecommended retained image range:")
    print(f"  Count {int(camera_window_start['Count'])} -> {int(camera_window_end['Count'])}  " )
       

    # Save pass metadata
    pass_info = pd.DataFrame(
        [{"Pass": pass_number,
          "DepositionStart":  deposition_start,
          "DepositionEnd": deposition_end,
          "DepositionDuration_s":  deposition_duration,
          "DwellDuration_s": DWELL_S,
          "DwellEnd": dwell_end,
          "WindowStart": window_start,
          "WindowEnd": window_end,
          "A700_DepositionStart_Count": int(camera_start["Count"]),
          "A700_DepositionEnd_Count": int(camera_deposition_end["Count"]),
          "A700_DwellEnd_Count": int(camera_dwell_end["Count"]),
          "A700_WindowStart_Count": int(camera_window_start["Count"]),
          "A700_WindowEnd_Count": int(camera_window_end["Count"])
        }]
    )


    pass_info.to_csv(pass_dir / f"Pass{pass_number}_Info.csv", index=False)

print("\nExperimental synchronization complete.")