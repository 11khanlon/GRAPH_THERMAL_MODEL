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


