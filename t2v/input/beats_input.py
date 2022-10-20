import logging
import typing

import numpy as np
from librosa.beat import beat_track
from omegaconf import DictConfig

from t2v.config.root import RootConfig
from t2v.input.input_mechanism import InputVariableMechanism
from t2v.input.spectral_input import read_audio_signed16


class BeatAudioParser(InputVariableMechanism):
    def __init__(self, config: DictConfig, root_config: RootConfig):
        super().__init__(config, root_config)
        input_audio = config["file"]
        offset = config["offset"]
        self.audio_samples = read_audio_signed16(input_audio).astype(np.float32) / 32767
        self.fps = root_config.frames_per_second
        logging.info(f"Getting beats for audio samples from {input_audio}...")
        self.beats = beat_track(y=self.audio_samples, sr=44100, units='time')
        logging.info(f"Detected BPM: {self.beats[0]}, beats found in audio at timestamps: {self.beats[1]}")
        self.offset = offset

    def is_beat(self, t):
        """
        Check if a beat happens at the given time or within the frame-timespan.

        :param t: time in seconds as float
        :return: True if a beat lies in the frame-timespan
        """
        # the frame-timespan ft is 1/fps seconds long.
        # We will fuzzy-match if there is a beat by checking if there is a beat within the [t-ft/2, t+ft/2] range.
        frame_timespan_half = (1 / self.fps) / 2
        ft_min = t + self.offset - frame_timespan_half
        ft_max = t + self.offset + frame_timespan_half
        if ft_min < 0:
            return False
        # TODO: this is pretty inefficent for now. binary search or something like that could be better
        # TODO we could return how well it matches instead of boolean (distance from actual beat)
        for beat in self.beats[1]:
            if ft_min < beat < ft_max:
                return True
        return False

    def func_var_callback(self, t) -> typing.Dict[str, float]:
        if self.is_beat(t):
            return {
                "beat": 1.0
            }
        else:
            return {
                "beat": 0.0
            }

    def prompt_modulator_callback(self, t) -> typing.Dict[str, str]:
        return super().prompt_modulator_callback(t)
