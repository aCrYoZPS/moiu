from dataclasses import dataclass

import numpy as np


@dataclass
class FlowNetwork:
    vertices: int
    capacities: np.ndarray