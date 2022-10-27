from dataclasses import dataclass
from typing import List

from omegaconf import DictConfig


@dataclass
class MechanismDefinition:
    type: str
    mechanism_parameters: DictConfig

@dataclass
class AdditionalContextConfig:
    input_mechanisms: List[MechanismDefinition]
