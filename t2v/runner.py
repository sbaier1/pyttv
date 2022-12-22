import csv
import logging
import math
import os
from datetime import timedelta
from types import ModuleType

from PIL import Image
from jinja2 import Template
from omegaconf import OmegaConf

from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig
from t2v.config.scene import Scene
from t2v.input.beats_input import BeatAudioParser
from t2v.input.midi_input import MidiInput
from t2v.input.spectral_input import SpectralAudioParser
from t2v.mechanism.api_mechanism import ApiMechanism
from t2v.mechanism.noop_mechanism import NoopMechanism
from t2v.util import parse_time

INTERPOLATE_DIRECTORY = "_interpolate"

mechanism_types = {
    NoopMechanism.name(): NoopMechanism,
    ApiMechanism.name(): ApiMechanism
}

reactivity_types = {
    "beats-librosa": BeatAudioParser,
    "spectral": SpectralAudioParser,
    "midi": MidiInput,
}

scene_progress = 0


class Runner:
    def __init__(self, cfg: RootConfig):
        # Register mechanisms
        self.func_util = FuncUtil(cfg)
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
        self.func_util.add_callback("runner_scene_progress", get_scene_progress)
        # TODO: ensure/create output dir
        self.initialize_additional_context()

    def run(self):
        logging.debug(f"Launching with config:\n{OmegaConf.to_yaml(self.cfg)}")
        if "simulate_output" in self.cfg:
            self.simulate_scenario()
            return
        # TODO: this is pretty stateful, when resuming a run the interpolation frames will not be used.
        #  Must find and load them in that case.
        interpolation_frames = []
        prev_prompt = None
        last_context = {}
        total_delta = timedelta()
        for i in range(0, len(self.cfg.scenes)):
            scene = self.cfg.scenes[i]
            cur_delta = parse_time(scene.duration)
            total_delta += cur_delta
            timestamp = f"{int(total_delta.total_seconds() / 60):02.0f}" \
                        f":{int(total_delta.total_seconds() % 60):02.0f}" \
                        f".{((total_delta.total_seconds() - int(total_delta.total_seconds())) * 1000):03.0f}"
            logging.info(
                f"Scene will run up to {timestamp} "
                f"(frame {int(total_delta.total_seconds() * self.cfg.frames_per_second):05.0f}) "
                f"for {cur_delta.total_seconds():02.02f}s with prompt {scene.prompt}")
        logging.info(f"Total duration of scenario: {total_delta.total_seconds():.3f}s")
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
                                                                   scene, frame_path, 1.0)
                        else:
                            mechanism.skip_frame()
                        interpolation_frames.append(frame_path)
                    prev_prompt = scene.prompt
                    mechanism.set_interpolation_state(interpolation_frames, prev_prompt)
                    mechanism.reset_scene_state()

    def simulate_scenario(self):
        logging.info(f"Running in simulation mode. Saving result to {self.cfg.simulate_output}")
        duration = 0
        for scene in self.cfg.scenes:
            self.get_or_initialize_mechanism(scene)
            duration += parse_time(scene.duration).total_seconds()
        frame_count = self.get_frame_count(duration)
        with open(self.cfg.simulate_output, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            # header
            func_map = self.func_util.update_math_env(0)
            for scene in self.cfg.scenes:
                mechanism = self.get_or_initialize_mechanism(scene)
                func_map.update(mechanism.simulate_step(scene.mechanism_parameters, 0))
            dict_keys = ["index", "t"]
            for key in func_map.keys():
                val = func_map[key]
                if not isinstance(val, ModuleType) \
                        and not hasattr(val, '__call__') \
                        and key not in ["t", "pi", "tau", "e", "__builtins__", "inf", "nan"]:
                    dict_keys.append(key)
            csv_writer.writerow(dict_keys)
            i = 0
            for scene in self.cfg.scenes:
                mechanism = self.get_or_initialize_mechanism(scene)
                k = 0
                while k < self.cfg.frames_per_second * parse_time(scene.duration).total_seconds():
                    value_row = []
                    t = i / self.cfg.frames_per_second
                    scene_progress = k / (self.cfg.frames_per_second * parse_time(scene.duration).total_seconds())
                    func_map = self.func_util.update_math_env(t)
                    func_map.update(mechanism.simulate_step(scene.mechanism_parameters, t))
                    func_map["index"] = i
                    for key in dict_keys:
                        if key in func_map:
                            val = func_map[key]
                            value_row.append(val)
                    csv_writer.writerow(value_row)
                    i += 1
                    k += 1

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
        step_in_scene = 0
        total_frames = parse_time(scene.duration).seconds * self.cfg.frames_per_second
        while (self.t - float(offset)) < parse_time(scene.duration).seconds:

            prev_frame_path = os.path.join(self.output_path, f"{self.frame - 1:05}.png")
            current_frame_path = os.path.join(self.output_path, f"{self.frame:05}.png")
            scene_progress = step_in_scene / total_frames
            if not os.path.exists(current_frame_path):
                logging.info(f"Rendering overall frame {self.frame} in scene with prompt {scene.prompt}, "
                             f"progress: {scene_progress}")
                if has_fast_forwarded:
                    # Inject prev frame
                    # noinspection PyTypeChecker
                    context["prev_image"] = Image.open(prev_frame_path)
                context = self.generate_and_save_frame(context, mechanism, scene,
                                                       current_frame_path, scene_progress)
            else:
                logging.info(f"Skipping frame {self.frame:05} because it already exists on disk")
                mechanism.skip_frame(scene.mechanism_parameters)
                has_fast_forwarded = True
            step_in_scene += 1

            self.frame = self.frame + 1
            self.t = (self.frame / self.cfg.frames_per_second)

        # If the scene has ended we're still going to inject this at the end
        # to make sure the interpolation logic will work if necessary
        if has_fast_forwarded:
            # Inject prev frame
            # noinspection PyTypeChecker
            context["prev_image"] = Image.open(prev_frame_path)
        return context

    def get_or_initialize_mechanism(self, scene):
        mechanism_name = scene.mechanism
        if self.mechanisms.get(mechanism_name) is None:
            mechanism = self._init_mechanism(mechanism_name)
        else:
            mechanism = self.mechanisms[mechanism_name]
        return mechanism

    def generate_and_save_frame(self, context, mechanism, scene, path, progress):
        template_dict = {'math': math,
                         'scene_progress': progress,
                         }
        # Add function context
        template_dict.update(self.func_util.update_math_env(self.t))
        evaluated_prompt = Template(scene.prompt).render(
            template_dict)

        def func(value):
            return ''.join(value.splitlines())

        evaluated_prompt = func(evaluated_prompt)
        logging.info(f"Evaluated prompt {evaluated_prompt}")
        image_frame, context = mechanism.generate(scene.mechanism_parameters, context, evaluated_prompt, self.t)
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
            input_mechanisms = self.cfg.additional_context.input_mechanisms
            types = {}
            for mechanism in input_mechanisms:
                logging.info("Initializing input mechanisms...")
                mechanism_type_name = mechanism.type
                cls = reactivity_types.get(mechanism_type_name)
                instance = cls(mechanism.mechanism_parameters, self.cfg)
                if mechanism_type_name not in types:
                    types[mechanism_type_name] = 1
                    self.func_util.add_callback(mechanism_type_name, instance.func_var_callback)
                else:
                    # Add suffix for multiple instances of the same type
                    self.func_util.add_callback(mechanism_type_name + f"_{types[mechanism_type_name] + 1}",
                                                instance.func_var_callback)
                    types[mechanism_type_name] += 1



# noinspection PyUnusedLocal
def get_scene_progress(t):
    return {
        'scene_progress': scene_progress
    }
