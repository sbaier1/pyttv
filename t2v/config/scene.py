from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass
class Scene:
    prompt: str
    duration: str
    interpolation: str
    mechanism: str
