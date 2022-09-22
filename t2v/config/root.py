from dataclasses import dataclass

from typing import List

from t2v.config.mechanism import MechanismDefinition
from t2v.config.scene import Scene


@dataclass
class RootConfig:
    """
    The root of the config file
    """
    frames_per_second: int
    scenes: List[Scene]
    mechanisms: List[MechanismDefinition]
    width: int = 512
    height: int = 512

