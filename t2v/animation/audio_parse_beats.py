import logging

import numpy as np
import typing
from librosa.beat import beat_track
from t2v.animation.audio_parse import read_audio_signed16


class BeatAudioParser:
    def __init__(self,
                 input_audio,
                 frames_per_second,
                 offset):
        self.audio_samples = read_audio_signed16(input_audio).astype(np.float32) / 32767
        self.fps = frames_per_second
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

    def get_params(self, t) -> typing.Dict[str, float]:
        if self.is_beat(t):
            return {
                "beat": 1.0
            }
        else:
            return {
                "beat": 0.0
            }
