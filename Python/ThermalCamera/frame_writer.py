
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from pathlib import Path
import numpy as np


@dataclass
class ThermalFrame:

    frame_id: int

    camera_timestamp: float | None

    host_timestamp: datetime

    raw: np.ndarray

    temperature: np.ndarray




class ThermalFrameWriter:

    def __init__(
        self,
        output_folder
    ):

        self.output_folder = Path(
            output_folder
        )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )


    def save_frame(
        self,
        frame
    ):

        frame_name = (
            f"{frame.frame_id:08d}"
        )

        np.save(
            self.output_folder
            / f"{frame_name}_raw.npy",

            frame.raw
        )

        np.save(
            self.output_folder
            / f"{frame_name}_temperature.npy",

            frame.temperature.astype(
                np.float32
            )
        )