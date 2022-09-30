import os
import typing

import PIL.Image
from PIL.Image import Image
from omegaconf import DictConfig

from t2v.animation.animator_3d import Animator3D
from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig


class Mechanism:
    """
    Defines an arbitrary text2video mechanism.
    """

    def __init__(self, config: DictConfig, root_config: RootConfig, func_util: FuncUtil):
        """
        Initialize the mechanism
        :param root_config: root config for mechanisms that depend on other global parameters
        :param config: config parameters that configure the mechanism
        :return: None
        """
        animation_type = config.get("animation")
        if animation_type is not None:
            if animation_type == "3D":
                self.animator = Animator3D(config.get("animation_parameters"), root_config, func_util)
            elif animation_type == "2D":
                # TODO 2D anim
                self.animator = None
        self.interpolation_frames = []
        self.interpolation_index = 0
        self.interpolation_prev_prompt = None

    def generate(self, config: DictConfig, context, prompt: str, t):
        """
        Generate a frame with the mechanism
        :param prompt: prompt to generate frames for
        :param config: configuration of the mechanism
        :param context: context from the previous generation, if any
        :return: the frame as a PIL image, and optionally the context for the next iteration
        """

    def destroy(self):
        """
        de-initialize the mechanism. Unload models / free up memory as much as possible
        :return:
        """

    def set_interpolation_state(self, interpolation_frames: typing.List[str], prev_prompt: str = None):
        """
        Set the interpolation state for interpolating from the previous to this scene.
        :param interpolation_frames: The frames for the duration of the interpolation that were rendered from the previous scene
        :param prev_prompt: The previous prompt if the interpolation implementation is text-conditioned
        """
        self.interpolation_frames = interpolation_frames
        self.interpolation_index = 0
        self.interpolation_prev_prompt = prev_prompt

    def reset_scene_state(self):
        """
        Marks the end of a scene to the mechanism. It can reset its scene-specific state here
        """

    @staticmethod
    def blend_frames(image1: Image, image2: Image, factor: float):
        """
        basic frame blending between two images
        :param image1: first image
        :param image2: second image
        :param factor: weighting, <= 0.5 favors im1, otherwise favors im2
        :return: new image
        """
        return PIL.Image.blend(image1, image2, factor)

    @staticmethod
    def name():
        """
        The name of the mechanism, by which the impl will be referenced in the config
        :return:
        """
