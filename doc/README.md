# Feature overview

## Parameter and prompt modulation

### Parameter modulation

Many parameters in pyttv can be modulated throughout the scenario using functions.

These functions are simply strings that are evaluated using python's `eval` method to retrieve their current value.

A base context for performing various mathematical operations is passed to these functions. See [here](https://github.com/sbaier1/pyttv/blob/67816a8be1aedf069cbf8bba43eff60c63423373/t2v/animation/func_tools.py#L10-L16) for a list of python modules and functions that are available.

Additionally, dynamic context is added to the function context, depending on your configuration

#### Dynamic context

* API mechanism: 
  * Adds a `is_turbo_step` variable that is either 1 or 0, depending on whether the current step is a turbo step.
* MIDI mechanism:
  * Adds a `notes` dict to the context which contains the `MIDI note number -> 1` mapping for the current frame. NOTE: At the moment, this is only `1` when the note_on event is happening in the current frame, therefore mainly useful for percussion tracks.
  * Adds a `note_playing(dict, int)` function to the context which returns 1 or 0, depending on whether the note with the given number is currently playing.
    * For example: `note_playing(notes, 36)` checks whether the MIDI note 36 (`C2`) is currently playing. See [here](https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies) for a full table of MIDI note numbers and corresponding notes.
    * hint: to you can use `note_playing` and multiplication to figure out whether multiple notes are currently playing.
* spectral mechanism
  * Adds a configurable variable name for each filter, which is in the `[0..1]` range.
* beats mechanism
  * Adds a `is_beat` variable that is either 1 or 0, depending on whether the current frame time is withing the range of a beat.

## Input mechanisms
TBD

### 
