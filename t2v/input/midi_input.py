import logging
import typing

import mido
from mido import tick2second, bpm2tempo
from omegaconf import DictConfig

from t2v.config.root import RootConfig
from t2v.input.input_mechanism import InputVariableMechanism
from t2v.input.midi_notes import name_to_note, name_to_note_no_oct


class MidiInput(InputVariableMechanism):
    def __init__(self, config: DictConfig, root_config: RootConfig):
        super().__init__(config, root_config)
        self.fps = root_config.frames_per_second
        midifile_location = config["file"]
        midifile = mido.MidiFile(midifile_location)
        self.prefix = ""
        if "prefix" in config:
            self.prefix = config["prefix"]
            logging.info(f"Using prefix {self.prefix} for midi input variables from file {midifile_location}")
        # default to using the first track in the file
        selected_midi_track = midifile.tracks[0]
        if "track" in config:
            track = config["track"]
            found = False
            for midi_track in midifile.tracks:
                if midi_track.name == track:
                    selected_midi_track = midi_track
                    logging.info(f"Using MIDI track with name {track} from file {midifile_location}")
                    found = True
            if not found:
                raise RuntimeError(f"Could not find MIDI track with name {track} in file {midifile_location}")
        tempo_override = None
        if "tempo" in config:
            tempo_override = config["tempo"]
            # Convert tempo in bpm to quarter node period in microseconds
            tempo_us = int(round((60 * 1000000) / tempo_override))
            selected_midi_track.insert(0, mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
        self.offset = 0 if "offset" not in config else config["offset"]
        offset = 0
        # Only tracks when notes are being triggered
        self.note_on_events = {}
        # Also tracks note state in time (i.e. is this note currently playing?)
        self.notes_playing = {}
        # Tracks notes that are currently playing
        currently_playing = {}
        tempo_us = bpm2tempo(120 if tempo_override is None else tempo_override)
        for msg in selected_midi_track:
            offset += tick2second(msg.time, midifile.ticks_per_beat, tempo_us)
            if msg.type == 'note_on':
                if offset in self.note_on_events:
                    self.note_on_events[offset][msg.note] = 1
                else:
                    self.note_on_events[offset] = {msg.note: 1}
                # Note tracking
                currently_playing[msg.note] = 1
                if offset not in self.notes_playing:
                    self.notes_playing[offset] = {}
                self.notes_playing[offset].update(currently_playing)
            if msg.type == 'note_off':
                # Condition because some midi files can be weird
                if msg.note in currently_playing:
                    del (currently_playing[msg.note])
        logging.info(f"Total duration of notes in midi file {config['file']}: {offset}s")

    def func_var_callback(self, t):
        # the frame-timespan ft is 1/fps seconds long.
        # We will fuzzy-match if there is a beat by checking if there is a beat within the [t-ft/2, t+ft/2] range.
        frame_timespan_half = (1 / self.fps) / 2
        ft_min = t + self.offset - frame_timespan_half
        ft_max = t + self.offset + frame_timespan_half
        res = {
            "note_playing": note_playing,
            f"{self.prefix}notes": {},
            f"{self.prefix}notes_currently_on": {},
        }
        if ft_min < 0:
            return res
        # TODO: this is pretty inefficent for now. binary search or something like that could be better
        # TODO we could return how well it matches instead of boolean (distance from actual beat)
        for note_time in self.note_on_events.keys():
            if ft_min < note_time <= ft_max:
                # noinspection PyTypeChecker
                res[f"{self.prefix}notes"].update(self.note_on_events[note_time])
        for note_time in self.notes_playing.keys():
            if ft_min < note_time <= ft_max:
                # noinspection PyTypeChecker
                res[f"{self.prefix}notes_currently_on"].update(self.notes_playing[note_time])
        return res

    def prompt_modulator_callback(self, t) -> typing.Dict[str, object]:
        return super().prompt_modulator_callback(t)


def note_playing(notes, note):
    """
    Helper function for testing if a note is being played when evaluating.
    When multiple notes are provided, this will return 1 if ANY of them is playing.
    If you want to check if multiple notes are playing, use multiple calls to this function and mathematical operations.
    :param notes: note dict
    :param note: note to check for
    :return: whether the note is being played in the given dict.
    """
    if isinstance(note, list):
        new_list = []
        # Ensure all notes are numbers first
        for inst in note:
            new_list.append(note_to_number(inst))
        for note in new_list:
            if note in notes:
                return 1
            else:
                return 0
    note = note_to_number(note)
    if note in notes:
        return 1
    else:
        return 0


def note_to_number(note):
    if isinstance(note, str):
        # convert to note number first
        if note in name_to_note:
            note = name_to_note[note]
        else:
            logging.warning(f"Could not find note with name {note} in note map. "
                            f"See midi_notes.py to see a full list of note names.")
    return note


def note_playing_any_oct(notes, note):
    """
    Checks if some (str type) note is playing, regardless of octave
    """
    note_playing(notes, name_to_note_no_oct[note])