from dataclasses import dataclass
from typing import List, Optional

from t2v.config.context import AdditionalContextConfig
from t2v.config.mechanism import MechanismDefinition
from t2v.config.scene import Scene


@dataclass
class RootConfig:
    """
    The root of the config file
    """
    frames_per_second: float
    scenes: List[Scene]
    mechanisms: List[MechanismDefinition]
    width: int
    height: int
    torch_device: str
    output_path: str
    persistence_dir: str
    simulate_output: Optional[str] = None
    additional_context: AdditionalContextConfig = None
