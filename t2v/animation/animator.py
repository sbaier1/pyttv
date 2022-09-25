import logging

from omegaconf import DictConfig

from t2v.animation.audio_parse import SpectralAudioParser
from t2v.config.root import RootConfig


class Animator:
    """
    An Animator is an extension for a text2image model that can transform a frame as input for the next generation
    in order to induce motion
    """

    def __init__(self, config: DictConfig, root_config: RootConfig):
        """
        Initialize the animator mechanism, etc.
        :type root_config: The config object from the root
        :type config: The animation_parameters config for the current scene to process
        """
        audio_input_file = config.get("input_audio_file")
        filters = config.get("input_audio_filters")
        if audio_input_file is not None and filters is not None and len(filters) > 0:
            logging.info(f"Initializing audio reactivity with input file {audio_input_file}...")
            self.audio_parser = SpectralAudioParser(audio_input_file, config.get("input_audio_offset", 0),
                                                    root_config.frames_per_second, filters)

    def apply(self, frame, prompt, context, t):
        """
        Apply the transformation of the animator to an input frame and return the output frame
        :param t:
        :param frame:
        :param prompt:
        :param context:
        :return:
        """

    def destroy(self, config):
        """
        Unload models, etc. and destroy this animator. It must not be reused after this method was invoked.
        :param config:
        :return:
        """
