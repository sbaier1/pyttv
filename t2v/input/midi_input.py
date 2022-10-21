import logging
import typing

import mido
from omegaconf import DictConfig

from t2v.config.root import RootConfig
from t2v.input.input_mechanism import InputVariableMechanism


class MidiInput(InputVariableMechanism):
    def __init__(self, config: DictConfig, root_config: RootConfig):
        super().__init__(config, root_config)
        self.fps = root_config.frames_per_second
        midifile = mido.MidiFile(config["file"])
        # TODO: allow arbitrary set_tempo events via config
        if "tempo" in config:
            tempo_override = config["tempo"]
            # Convert tempo in bpm to quarter node period in microseconds
            tempo_us = int(60 / tempo_override * 1_000_000)
            midifile.tracks[0].insert(0, mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
        self.offset = 0 if "offset" not in config else config["offset"]
        offset = 0
        self.note_events = {}
        # TODO: separate by midi track and prefix with sanitized name
        # TODO: print some info for the user to be able to handle these easier
        for msg in midifile:
            offset += msg.time
            if msg.type == 'note_on':
                if offset in self.note_events:
                    self.note_events[offset][msg.note] = 1
                else:
                    self.note_events[offset] = {msg.note: 1}
        logging.info(f"Total duration of notes in midi file {config['file']}: {offset}s")

    def func_var_callback(self, t):
        # the frame-timespan ft is 1/fps seconds long.
        # We will fuzzy-match if there is a beat by checking if there is a beat within the [t-ft/2, t+ft/2] range.
        frame_timespan_half = (1 / self.fps) / 2
        ft_min = t + self.offset - frame_timespan_half
        ft_max = t + self.offset + frame_timespan_half
        res = {
            "note_playing": note_playing,
            "notes": {},
        }
        if ft_min < 0:
            return res
        # TODO: this is pretty inefficent for now. binary search or something like that could be better
        # TODO we could return how well it matches instead of boolean (distance from actual beat)
        for note_time in self.note_events.keys():
            if ft_min < note_time < ft_max:
                # noinspection PyTypeChecker
                res["notes"] = self.note_events[note_time]
                # FIXME: actually have to keep searching in case other events happen in this timeframe
                return res
        return res

    def prompt_modulator_callback(self, t) -> typing.Dict[str, str]:
        return super().prompt_modulator_callback(t)


# Helper function for testing if a note is being played when evaluating
def note_playing(notes, note):
    if note in notes:
        return 1
    else:
        return 0
