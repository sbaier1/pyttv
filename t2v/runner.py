import logging
import os
import re
from datetime import timedelta

import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from t2v.animation.audio_parse import SpectralAudioParser
from t2v.animation.audio_parse_beats import BeatAudioParser
from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig
from t2v.config.scene import Scene
from t2v.mechanism.api_mechanism import ApiMechanism
from t2v.mechanism.noop_mechanism import NoopMechanism
from t2v.mechanism.turbo_stablediff_mechanism import TurboStableDiff

INTERPOLATE_DIRECTORY = "_interpolate"

time_regex = re.compile(
    r'(?=((?P<milliseconds>\d+?)ms)?)(?=((?P<hours>\d+?)hr)?)(?=((?P<minutes>\d+?)m(?!s))?)(?=((?P<seconds>\d+?)s)?)')

TYPE_SPECTRAL = "spectral"
TYPE_LIBROSA = "beats-librosa"

mechanism_types = {
    TurboStableDiff.name(): TurboStableDiff,
    NoopMechanism.name(): NoopMechanism,
    ApiMechanism.name(): ApiMechanism
}


class Runner:
    def __init__(self, cfg: RootConfig):
        # Register mechanisms
        self.func_util = FuncUtil()
        self.cfg = cfg
        # Stores instantiated mechanism objects by their name within the config
        self.mechanisms = {}
        path = self.cfg.output_path
        self.output_path = path
        os.makedirs(path, exist_ok=True)
        # Interpolation frames subdirectory
        os.makedirs(os.path.join(path, INTERPOLATE_DIRECTORY), exist_ok=True)
        offset = 0
        self.frame = 0
        self.t = float(0)
        self.scene_offset = 0
        # TODO: ensure/create output dir
        self.initialize_additional_context()

    def run(self):
        logging.debug(f"Launching with config:\n{OmegaConf.to_yaml(self.cfg)}")
        # TODO: this is pretty stateful, when resuming a run the interpolation frames will not be used.
        #  Must find and load them in that case.
        interpolation_frames = []
        prev_prompt = None
        last_context = {}
        for i in range(self.scene_offset, len(self.cfg.scenes)):
            scene = self.cfg.scenes[i]
            logging.info(f"Rendering scene with prompt {scene.prompt}")
            last_context = self.handle_scene(scene, self.t, interpolation_frames, prev_prompt, last_context)
            # Remove interpolation frames from prev scene, if any
            interpolation_frames = []
            # Get interpolation frames, if any
            if i < len(self.cfg.scenes) - 1:
                interpolation_duration = parse_time(self.cfg.scenes[i + 1].interpolation)
                context = last_context
                if i < len(self.cfg.scenes) - 1 and interpolation_duration.total_seconds() > 0:
                    frame_count = self.get_frame_count(interpolation_duration.total_seconds())
                    mechanism = self.get_or_initialize_mechanism(scene)
                    for k in range(0, frame_count):
                        # Get the interpolation frames:
                        # These are additional, initially unused frames from the current scene.
                        frame_path = os.path.join(self.output_path, INTERPOLATE_DIRECTORY, f"{i:02}_{k:05}.png")
                        if not os.path.exists(frame_path):
                            context = self.generate_and_save_frame(context, mechanism,
                                                                   scene, frame_path)
                        else:
                            mechanism.skip_frame()
                        interpolation_frames.append(frame_path)
                    prev_prompt = scene.prompt
                    mechanism.set_interpolation_state(interpolation_frames, prev_prompt)
                    mechanism.reset_scene_state()

    def get_frame_count(self, duration: float):
        """
        Get number of frames that covers the given duration. Obviously comes with some inaccuracy.
        :param duration: duration in seconds, can have millisecond accuracy
        :return: number of frames
        """
        frame_seconds = 1 / self.cfg.frames_per_second
        return int(duration / frame_seconds)

    def handle_scene(self, scene: Scene, offset,
                     interpolation_frames, prev_prompt=None, init_context={}):
        mechanism = self.get_or_initialize_mechanism(scene)
        context = init_context
        prev_frame_path = None
        has_fast_forwarded = False
        while (self.t - float(offset)) < parse_time(scene.duration).seconds:

            prev_frame_path = os.path.join(self.output_path, f"{self.frame - 1:05}.png")
            current_frame_path = os.path.join(self.output_path, f"{self.frame:05}.png")
            if not os.path.exists(current_frame_path):
                logging.info(f"Rendering overall frame {self.frame} in scene with prompt {scene.prompt}")
                if has_fast_forwarded:
                    # Inject prev frame
                    # noinspection PyTypeChecker
                    context["prev_image"] = np.array(Image.open(prev_frame_path)).astype(np.uint8)
                context = self.generate_and_save_frame(context, mechanism, scene,
                                                       current_frame_path)
            else:
                logging.info(f"Winding past frame {self.frame:05} because it already exists on disk")
                mechanism.skip_frame()
                has_fast_forwarded = True

            self.frame = self.frame + 1
            self.t = (self.frame / self.cfg.frames_per_second)

        # If the scene has ended we're still going to inject this at the end
        # to make sure the interpolation logic will work if necessary
        if has_fast_forwarded:
            # Inject prev frame
            # noinspection PyTypeChecker
            context["prev_image"] = np.array(Image.open(prev_frame_path)).astype(np.uint8)
        return context

    def get_or_initialize_mechanism(self, scene):
        mechanism_name = scene.mechanism
        if self.mechanisms.get(mechanism_name) is None:
            mechanism = self._init_mechanism(mechanism_name)
        else:
            mechanism = self.mechanisms[mechanism_name]
        return mechanism

    def generate_and_save_frame(self, context, mechanism, scene, path):
        image_frame, context = mechanism.generate(scene.mechanism_parameters, context, scene.prompt, self.t)
        image_frame.save(path)
        return context

    def _init_mechanism(self, mechanism_name):
        # instantiate the mechanism
        mechanism_config = self._get_mechanism_config(mechanism_name)
        if mechanism_config is None:
            raise RuntimeError(f"Mechanism {mechanism_name} is not defined in the config")
        cls = mechanism_types.get(mechanism_config.type)
        if cls is None:
            raise RuntimeError(f"Mechanism type {mechanism_config.type} is not implemented")
        # instantiate the impl class
        logging.info(f"Initializing mechanism {mechanism_name} of type {mechanism_config.type}")
        mechanism = cls(mechanism_config.mechanism_parameters, self.cfg, self.func_util)
        self.mechanisms[mechanism_name] = mechanism
        return mechanism

    def _get_mechanism_config(self, name):
        for mechanism in self.cfg.mechanisms:
            if mechanism.name == name:
                return mechanism
        return None

    def initialize_additional_context(self):
        if "additional_context" in self.cfg and self.cfg.additional_context is not None:
            reactivity_config = self.cfg.additional_context.audio_reactivity
            if reactivity_config is not None:
                logging.info("Initializing audio reactivity...")
                audio_input_file = reactivity_config.input_audio_file
                audio_reactivity_type = reactivity_config.type
                filters = reactivity_config.input_audio_filters
                if audio_input_file is not None and audio_reactivity_type is not None:
                    if audio_reactivity_type == TYPE_LIBROSA:
                        logging.info(f"Initializing beat audio reactivity with input file {audio_input_file}...")
                        audio_parser = BeatAudioParser(audio_input_file, self.cfg.frames_per_second,
                                                       reactivity_config.input_audio_offset)
                        self.func_util.add_callback("audio", audio_parser.get_params)
                    elif audio_reactivity_type == TYPE_SPECTRAL and filters is not None and len(filters) > 0:
                        logging.info(f"Initializing audio reactivity with input file {audio_input_file}...")
                        audio_parser = SpectralAudioParser(audio_input_file, reactivity_config.input_audio_offset,
                                                           self.cfg.frames_per_second, filters)
                        self.func_util.add_callback("audio", audio_parser.get_params)
                    else:
                        logging.error(f"Could not initialize audio-reactivity, missing parameters or invalid type? "
                                      f"Valid types: {TYPE_LIBROSA}, {TYPE_SPECTRAL}")


def parse_time(time_str):
    parts = time_regex.match(time_str)
    if not parts:
        return
    parts = parts.groupdict()
    time_params = {}
    for name, param in parts.items():
        if param:
            time_params[name] = int(param)
    return timedelta(**time_params)
