from dataclasses import dataclass
from typing import List


@dataclass
class SpectralAudioFilter:
    variable_name: str
    f_center: int
    f_width: int
    order: int

@dataclass
class AudioReactivityConfig:
    type: str
    input_audio_file: str
    input_audio_offset: float
    input_audio_filters: List[SpectralAudioFilter]

@dataclass
class AdditionalContextConfig:
    audio_reactivity: AudioReactivityConfig

