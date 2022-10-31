from dataclasses import dataclass
from typing import List

from omegaconf import DictConfig


@dataclass
class MechanismDefinition:
    type: str
    mechanism_parameters: DictConfig


@dataclass
class FunctionDefinition:
    variable_name: str
    function: str
    prev_values: int = 0


@dataclass
class AdditionalContextConfig:
    custom_functions: List[FunctionDefinition]
    input_mechanisms: List[MechanismDefinition]
