import logging
import os

import cv2
import numpy as np
from PIL import Image
from omegaconf import DictConfig

from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig
from t2v.mechanism.mechanism import Mechanism
from t2v.mechanism.turbo_stablediff_functions import sample_to_cv2, maintain_colors, sample_from_cv2, add_noise


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
        # Some additional interpolation state
        self.interpolation_strength_history = []

    def generate(self, config: DictConfig, context, prompt: str, t):
        super().generate(config, context, prompt, t)
        # Overlay config
        merged_config = dict(self.config.copy())
        if config is not None:
            merged_config.update(config)

        debug = False
        if "debug" in merged_config:
            debug = merged_config["debug"]

        strength_evaluated = self.func_util.parametric_eval(merged_config.get("strength_schedule"), t)
        # common Img2Img pipeline
        interpolation_end = False
        if "prev_image" in context or len(self.interpolation_frames) > 0:
            if "prev_image" in context:
                previous_image = context["prev_image"]
            else:
                previous_image = None
            # Interpolate
            previous_image, strength_evaluated, interpolation_end = self.interpolate(merged_config, previous_image,
                                                                                     strength_evaluated, t)
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
            else:
                context["wrapped_context"] = {
                    "prev_image": noised_image
                }
        self.index = self.index + 1

        # Call wrapped model to generate the next frame
        if "wrapped_context" in context:
            context["wrapped_context"]["strength"] = strength_evaluated
            # Mark interpolation end for mechanisms that want to handle this state
            context["wrapped_context"]["interpolation_end"] = interpolation_end
            context["wrapped_context"]["interpolation_ongoing"] = self.interpolation_ongoing
            image, context = self.mechanism_callback(merged_config, context["wrapped_context"], prompt, t)
        else:
            image, context = self.mechanism_callback(merged_config, {"strength": strength_evaluated}, prompt, t)

        if self.color_match_sample is None and len(self.interpolation_frames) == 0:
            self.color_match_sample = np.array(image)
        return image, {
            "prev_image": image,
            "wrapped_context": context
        }

    def skip_frame(self):
        self.index = self.index + 1
        if self.interpolation_ongoing \
                and len(self.interpolation_frames) > 0 \
                and self.interpolation_index < len(self.interpolation_frames):
            self.interpolation_index = self.interpolation_index + 1
        elif self.interpolation_ongoing and self.interpolation_index == len(self.interpolation_frames):
            self.stop_interpolation()

    def destroy(self):
        super().destroy()

    def reset_scene_state(self):
        self.color_match_sample = None

    @staticmethod
    def name():
        # Not meant for direct instantiation
        return None

    def interpolate(self, config, previous_image, strength_evaluated, t):
        """
        Naive/generic interpolation (t2i approach-agnostic):
        * generate extra frames from prev. scene for the interpolation duration
        * blend the frames with the previous scene in a slope when generating the next one
        * disable color matching during transition
        * optionally increase the denoising strength in the same slope during the process to improve the transition
        """
        interpolation_ended = False
        if self.interpolation_ongoing and len(self.interpolation_frames) > 0 and self.interpolation_index < len(
                self.interpolation_frames):
            # Disable color matching during the interpolation, so we don't force-keep the previous scene color profile
            # TODO: make color matching more progressive instead of on/off.
            #  maybe progressively reduce its weight instead?
            self.color_match_sample = None
            interpolation_frame = self.interpolation_frames[self.interpolation_index]
            interpolation_function = config.get("interpolation_function")
            percentage = float((self.interpolation_index + 1) / len(self.interpolation_frames))
            if interpolation_function is not None:
                # 0..1 percentage how far along the interpolation is
                factor = self.func_util.parametric_eval(interpolation_function, t, x=percentage)
            else:
                # linear interpolation
                factor = percentage
            if "init_frame" in config:
                previous_image = Image.open(config.get("init_frame"))
            # Set the result image of the blend as the input for the ongoing animation
            if previous_image is not None:
                previous_image = np.asarray(
                    self.blend_frames(Image.open(interpolation_frame), previous_image, factor))
            else:
                raise RuntimeError("Interpolations must always have a previous image")
            # modulate the denoising strength while the interpolation is ongoing to retain more of the interpolation frames at the start, then make sure we reach 0 strength at the end
            strength_evaluated_prev = strength_evaluated
            if not self.interpolation_transition_complete:
                strength_evaluated = min(0.9, max(0.1, strength_evaluated + (1 - ((factor ** 1.4) * 1.65))))
                logging.info(
                    f"strength modulation: {strength_evaluated_prev} -> {strength_evaluated}. "
                    f"diff: {abs(strength_evaluated - strength_evaluated_prev)}, at evaluated percentage {factor}")
                self.interpolation_strength_history.append(strength_evaluated)
            if strength_evaluated < 0.25 \
                    or np.average(np.array(self.interpolation_strength_history)
                                  * np.linspace(0, 1, len(self.interpolation_strength_history)) ** 0.5) < 0.4:
                # Stop modulating strength if the current step or
                # the average of recent steps' strength is fairly low,
                # just assume we finished the transition
                logging.info(f"Considering transition as complete, stopping strength modulation, "
                             f"history size: {len(self.interpolation_strength_history)}")
                self.interpolation_transition_complete = True
            self.interpolation_index = self.interpolation_index + 1
        elif self.interpolation_ongoing and self.interpolation_index == len(self.interpolation_frames):
            self.stop_interpolation()
            interpolation_ended = True
            self.interpolation_strength_history = []
        return previous_image, strength_evaluated, interpolation_ended

    def stop_interpolation(self):
        # Interpolation finished, mark end, ensure this doesn't get called again
        self.interpolation_index = len(self.interpolation_frames) + 1
        self.interpolation_frames = []
        self.interpolation_prev_prompt = None
        # Reset color matching again so we can start over fresh with the new scene now
        self.color_match_sample = None
