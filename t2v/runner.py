import logging
import os
import re

from omegaconf import OmegaConf

from t2v.animation.animator_3d import Animator3D
from t2v.config.root import RootConfig
from t2v.config.scene import Scene
from t2v.mechanism.noop_mechanism import NoopMechanism
from t2v.mechanism.turbo_stablediff_mechanism import TurboStableDiff

from datetime import timedelta

time_regex = re.compile(r'((?P<hours>\d+?)hr)?((?P<minutes>\d+?)m)?((?P<seconds>\d+?)s)?')


class Runner:
    def __init__(self, cfg: RootConfig):
        # Register mechanisms
        self.mechanism_types = {
            TurboStableDiff.name(): TurboStableDiff,
            NoopMechanism.name(): NoopMechanism,
        }
        self.cfg = cfg
        # Stores instantiated mechanism objects by their name within the config
        self.mechanisms = {}
        path = self.cfg.output_path
        self.output_path = path
        os.makedirs(path, exist_ok=True)
        offset = 0
        for dirpath, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                # TODO: stupid resume counting for now. Match the filename pattern also
                if fname.endswith(".png"):
                    offset = offset + 1
        if offset > 0:
            self.frame = offset
            self.t = float(offset) / float(self.cfg.frames_per_second)
            t_buffer = 0
            scene_idx = 0
            for scene in cfg.scenes:
                scene_duration_seconds = parse_time(scene.duration).seconds
                if scene_duration_seconds > self.t + t_buffer:
                    # This is the scene we're resuming
                    break
                else:
                    t_buffer = t_buffer + scene_duration_seconds
                    scene_idx = scene_idx + 1
            self.scene_offset = scene_idx
            # TODO load the last frame as init?
        else:
            self.frame = 0
            self.t = float(0)
            self.scene_offset = 0
        # TODO: ensure/create output dir

    def run(self):
        logging.debug(f"Launching with config:\n{OmegaConf.to_yaml(self.cfg)}")
        for i in range(self.scene_offset, len(self.cfg.scenes)):
            scene = self.cfg.scenes[i]
            logging.info(f"Rendering scene with prompt {scene.prompt}")
            self.handle_scene(scene, self.t)

    def handle_scene(self, scene: Scene, offset):
        mechanism_name = scene.mechanism
        if self.mechanisms.get(mechanism_name) is None:
            mechanism = self._init_mechanism(mechanism_name)
        else:
            mechanism = self.mechanisms[mechanism_name]
        context = {}
        while (self.t - float(offset)) < parse_time(scene.duration).seconds:
            logging.debug(f"Rendering overall frame {self.frame} in scene with prompt {scene.prompt}")
            image_frame, context = mechanism.generate(scene.mechanism_parameters, context, scene.prompt, self.t)
            image_frame.save(os.path.join(self.output_path, f"{self.frame:05}.png"))
            # TODO: write frame to disk
            self.frame = self.frame + 1
            self.t = (self.frame / self.cfg.frames_per_second)

    def _init_mechanism(self, mechanism_name):
        # instantiate the mechanism
        mechanism_config = self._get_mechanism_config(mechanism_name)
        if mechanism_config is None:
            raise RuntimeError(f"Mechanism {mechanism_name} is not defined in the config")
        cls = self.mechanism_types.get(mechanism_config.type)
        if cls is None:
            raise RuntimeError(f"Mechanism type {mechanism_config.type} is not implemented")
        # instantiate the impl class
        logging.info(f"Initializing mechanism {mechanism_name} of type {mechanism_config.type}")
        mechanism = cls(mechanism_config.mechanism_parameters, self.cfg)
        self.mechanisms[mechanism_name] = mechanism
        return mechanism

    def _get_mechanism_config(self, name):
        for mechanism in self.cfg.mechanisms:
            if mechanism.name == name:
                return mechanism
        return None


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
