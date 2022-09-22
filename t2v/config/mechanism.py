from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class MechanismDefinition:
    name: str
    type: str
    mechanism_parameters: DictConfig
