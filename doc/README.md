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
  * Adds a `notes` dict to the context which contains the `MIDI note number -> 1` mapping for the current frame. NOTE: this dict only contains note_on events at the current point in time. This is useful specifically for reacting to percussion.
  * Adds a `notes_currently_on` dict to the context which contains the same mapping for the current frame. NOTE: this dict contains notes that have previously had note_on events and not yet note_off events. Essentially, this is the correct way for tracking notes for harmonic analysis.
  * Adds a `note_playing(dict, int|str)` function to the context which returns 1 or 0, depending on whether the note with the given number (or octave-specific name) is currently playing.
  * Adds a `note_playing_any_oct(dict, str)` function to the context which returns 1 or 0, depending on whether the note (name without octave, e.g. `C` or `Ab`) is playing
    * For example: `note_playing(notes, 36)` checks whether the MIDI note 36 (`C2`) is currently playing. See [here](https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies) for a full table of MIDI note numbers and corresponding notes.
    * hint: you can also use the natural "english note names"(capitalized) for checking whether a note is being played.
    * hint: to you can use `note_playing` and multiplication to figure out whether multiple notes are currently playing.
* spectral mechanism
  * Adds a configurable variable name for each filter, which is in the `[0..1]` range.
* beats mechanism
  * Adds a `is_beat` variable that is either 1 or 0, depending on whether the current frame time is withing the range of a beat.
* runner (always available)
  * Adds a `scene_progress` variable that tracks how far along the animation is within the current scene in the `[0..1]` range. 
* custom_functions of course add arbitrary variables to the context as well.

#### Custom variables/functions

You can also add custom variables that are derived from functions. The variables can then be used in other functions.

Unlike other functions, these functions can also be stateful, as in the previous values are stored in a list.
This allows for more advanced functions like adding smoothing or slow decay to input variables. 

## Input mechanisms

### MIDI mechanism

The MIDI mechanism reads MIDI files and adds variables to the dynamic context which track the note status along with the animation.

hint: You can use tools like omnizart, spleeter and melodyne to generate MIDI files for existing songs. These can be a bit hit or miss though. If you are lucky, someone else already transcribed your song and provided a full MIDI file with tracks for your song.
