import logging

from omegaconf import OmegaConf

from t2v.config.root import RootConfig
from t2v.config.scene import Scene
from t2v.mechanism.noop_mechanism import NoopMechanism
from t2v.mechanism.turbo_stablediff_mechanism import TurboStableDiff


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
        # TODO: resume runs must determine the frame, t-offset and scene offset here
        self.t = float(0)

    def run(self):
        logging.debug(f"Launching with config:\n{OmegaConf.to_yaml(self.cfg)}")
        for scene in self.cfg.scenes:
            self.handle_scene(scene)

    def handle_scene(self, scene: Scene):
        mechanism_name = scene.mechanism
        if self.mechanisms.get(mechanism_name) is None:
            mechanism = self._init_mechanism(mechanism_name)
        else:
            mechanism = self.mechanisms[mechanism_name]


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
        mechanism = cls()
        self.mechanisms[mechanism_name] = mechanism
        mechanism.init(mechanism_config.mechanism_parameters)
        return mechanism

    def _get_mechanism_config(self, name):
        for mechanism in self.cfg.mechanisms:
            if mechanism.name == name:
                return mechanism
        return None
