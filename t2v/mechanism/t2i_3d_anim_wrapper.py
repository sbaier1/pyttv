import os
import typing

import numpy as np
from PIL.Image import Image
from omegaconf import DictConfig

from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig

from t2v.mechanism.mechanism import Mechanism
from t2v.mechanism.turbo_stablediff_functions import sample_to_cv2, maintain_colors, sample_from_cv2, add_noise

import cv2


class T2IAnimatedWrapper(Mechanism):
    """
    A wrapper containing common logic for
    - 3d warping images from a text2img/img2img mechanism
    - simple interpolating between scenes for img2img by blending frames
    - color matching which adapts to scene change interpolations

    For composing other mechanisms that use these paradigms.

    Adds a "prev_image" key to the context of the wrapped mechanism,
    which contains the warped previous frame as uint8 numpy array.
    """

    def __init__(self, config: DictConfig, root_config: RootConfig, func_util: FuncUtil,
                 mechanism_callback: Mechanism.generate, mechanism: Mechanism):
        super().__init__(config, root_config, func_util)
        self.func_util = func_util
        self.config = config
        self.root_config = root_config
        self.mechanism_callback = mechanism_callback
        self.mechanism = mechanism
        self.color_match_sample = None
        self.index = 0

    def generate(self, config: DictConfig, context, prompt: str, t):
        super().generate(config, context, prompt, t)
        # Overlay config
        merged_config = self.config.copy()
        if config is not None:
            merged_config.update(config)

        debug = False
        if "debug" in merged_config:
            debug = merged_config["debug"]

        strength_evaluated = self.func_util.parametric_eval(merged_config.get("strength_schedule"), t)
        # common Img2Img pipeline
        if "prev_image" in context or len(self.interpolation_frames) > 0:
            previous_image = context["prev_image"]
            # Interpolate
            previous_image, strength_evaluated = self.interpolate(merged_config, previous_image, strength_evaluated, t)
            # Warp
            warped_frame = self.animator.apply(np.array(previous_image), prompt,
                                               merged_config.get("animation_parameters"), t)
            if debug:
                cv2.imwrite(os.path.join(self.root_config.output_path, f"{self.index:05}_warped.png"),
                            cv2.cvtColor(warped_frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
            # Color match
            if self.color_match_sample is not None:
                warped_frame = maintain_colors(warped_frame, self.color_match_sample, 'Match Frame 0 LAB')

            # TODO: parameters for contrast schedule, noise schedule
            # apply scaling
            # contrast_sample = warped_frame * 0.95

            # Frame noising
            noised_sample = add_noise(sample_from_cv2(warped_frame),
                                      self.func_util.parametric_eval(merged_config.get("noise_schedule"), t))
            # Convert back to image
            noised_image = sample_to_cv2(noised_sample)
            if "wrapped_context" in context:
                context["wrapped_context"]["prev_image"] = noised_image
        self.index = self.index + 1

        # Call wrapped model to generate the next frame
        if "wrapped_context" in context:
            image, context = self.mechanism_callback(merged_config, context["wrapped_context"], prompt, t)
        else:
            image, context = self.mechanism_callback(merged_config, {}, prompt, t)

        if self.color_match_sample is None:
            self.color_match_sample = np.array(image)
        return image, {
            "prev_image": image,
            "wrapped_context": context
        }

    def destroy(self):
        super().destroy()

    def reset_scene_state(self):
        self.color_match_sample = None

    @staticmethod
    def name():
        # Not meant for direct instantiation
        return None

    def interpolate(self, config, previous_image, strength_evaluated, t):
        # TODO: this naive image blending doesn't work very well yet.
        #        - does it make sense / is it possible to weighted-condition the prompt on the previous one?
        #        - can we have multiple init samples for img2img during the interpolation to make them more alike? can those be weighted?
        #        - does it make sense to have a sloped denoising reduction during the interpolation to keep more of the interpolation frames?
        if len(self.interpolation_frames) > 0 and self.interpolation_index < len(self.interpolation_frames):
            # Disable color matching during the interpolation, so we don't force-keep the previous scene color profile
            self.color_match_sample = None
            interpolation_frame = self.interpolation_frames[self.interpolation_index]
            interpolation_function = config.get("interpolation_function")
            percentage = float(self.interpolation_index / len(self.interpolation_frames))
            if interpolation_function is not None:
                # 0..1 percentage how far along the interpolation is
                factor = self.func_util.parametric_eval(interpolation_function, t, x=percentage)
            else:
                # linear interpolation
                factor = percentage
            # Set the result image of the blend as the input for the ongoing animation
            if previous_image is not None:
                previous_image = np.asarray(
                    self.blend_frames(Image.open(interpolation_frame), previous_image, factor))
            else:
                previous_image = np.asarray(Image.open(interpolation_frame))
            # modulate the denoising strength while the interpolation is ongoing to retain more of the interpolation frames
            # the 1.5 factor ensures we go to the minimum clamped strength so a full transition to the new scene can be
            # made without retaining some features of the previous scene forever.
            strength_evaluated = min(1.0, max(0.1, strength_evaluated + ((1 - (factor * 1.5)) * 0.6)))
            self.interpolation_index = self.interpolation_index + 1
        elif self.interpolation_index == len(self.interpolation_frames):
            # Interpolation finished, mark end, ensure this doesn't get called again
            self.interpolation_index = len(self.interpolation_frames) + 1
            self.interpolation_frames = []
            self.interpolation_prev_prompt = None
            # Reset color matching again so we can start over fresh with the new scene now
            self.color_match_sample = None
            # Set the strength very low intentionally so the new scene can properly influence the image now and we don't retain too much over time
            return previous_image, 0.2
        return previous_image, strength_evaluated
