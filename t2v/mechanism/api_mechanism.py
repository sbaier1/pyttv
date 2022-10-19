import base64
import io
import logging
import typing

import numpy as np
from PIL import Image
from omegaconf import DictConfig

from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig

from t2v.mechanism.mechanism import Mechanism

import requests

from t2v.mechanism.t2i_3d_anim_wrapper import T2IAnimatedWrapper
# Generated with revision 604620a7f08d1126a8689f9f4bec8ade0801a69b,
# diff the template with a captured query of a more recent version to update these when necessary
from t2v.mechanism.turbo_stablediff_functions import add_noise, sample_from_cv2, sample_to_cv2, maintain_colors

TXT2IMG = """
{{
"fn_index": 13,
  "data": [
    "{prompt}",
    "",
    "None",
    "None",
    {steps},
    "{sampler}",
    false,
    false,
    1,
    1,
    {scale},
    {seed},
    -1,
    0,
    0,
    0,
    false,
    {H},
    {W},
    {hires_fix_enabled},
    {hires_denoising_strength},
    {W_init},
    {H_init},
    "None",
    false,
    false,
    null,
    "",
    "Seed",
    "",
    "Nothing",
    "",
    true,
    false,
    false,
    null,
    ""
  ],
  "session_hash": "djrqd1giwif"
}}
"""

IMG2IMG = """
{{
"fn_index": 33,
    "data": [
        0,
        "{prompt}",
        "",
        "None",
        "None",
        "data:image/png;base64,{pngbase64}",
        null,
        null,
        null,
        "Draw mask",
        {steps},
        "{sampler}",
        4,
        "fill",
        false,
        false,
        1,
        1,
        {scale},
        {strength},
        {seed},
        -1,
        0,
        0,
        0,
        false,
        {H},
        {W},
        "Just resize",
        false,
        32,
        "Inpaint masked",
        "",
        "",
        "None",
        "",
        true,
        true,
        "",
        "",
        true,
        50,
        true,
        1,
        0,
        false,
        4,
        1,
        "",
        128,
        8,
        [
            "left",
            "right",
            "up",
            "down"
        ],
        1,
        0.05,
        128,
        4,
        "fill",
        [
            "left",
            "right",
            "up",
            "down"
        ],
        "",    
        true,
        true,
        "",
        "",
        true,
        50,
        true,
        1,
        0,
        false,
        false,
        false,
        null,
        "",
        "",
        64,
        "None",
        "Seed",
        "",
        "Nothing",
        "",
        true,
        false,
        false,
        null,
        "",
        ""
],
"session_hash": "djrqd1giwif"
}}"""


class ApiMechanism(Mechanism):
    def __init__(self, config: DictConfig, root_config: RootConfig, func_util: FuncUtil):
        super().__init__(config, root_config, func_util)
        self.root_config = root_config
        self.config = config
        self.func_util = func_util
        func_util.add_callback("isTurboStep", self.is_turbo_step)
        self.host = self.config.get("host")
        self.index = 0
        self.anim_wrapper = T2IAnimatedWrapper(config, root_config, func_util, self.actual_generate, self)
        self.scene_init = True

    def is_turbo_step(self, t):
        if self.index % (self.config["turbo_steps"] + 1) == 0:
            return {"is_turbo_step": 0}
        else:
            return {"is_turbo_step": 1}

    def generate(self, config: DictConfig, context, prompt: str, t):
        return self.anim_wrapper.generate(config, context, prompt, t)

    def skip_frame(self):
        self.index = self.index + 1

    def actual_generate(self, config: DictConfig, context, prompt: str, t):
        # TODO: template out the queries, run txt2img, decode the image
        # TODO: on subsequent steps, run img2img, encode the previous frame as base64 and use as input
        # TODO: warping code from other mechanism
        # Start with the root (default) config
        config_copy = dict(self.config.copy())
        # Overlay the scene-specific params if necessary
        if config is not None:
            config_copy.update(config)
        if "strength" not in context:
            config_copy.update({"strength": 1 - self.func_util.parametric_eval(config.get("strength_schedule"), t)})
        else:
            # Invert input strength because it works the other way round in this mechanism
            config_copy.update({"strength": 1 - context['strength']})

        # TODO: key latents don't work here atm. img2img is too different from txt2img with scaled latents.
        #   idea: implement img2img module for automatic1111 for slerping between prompts, use that to generate all interpolation frames and write them directly.
        #   idea(easier?): generate the key latent image first, then run img2img between prev prompt with down-sloping denoising (denoise 0 in last step to finish transition)
        if "interpolation_ongoing" in context and context["interpolation_ongoing"] and "seed" in config:
            # keep index at 0 during the interpolation in this case to make sure we interpolate towards that desired frame
            self.index = 0
        if "interpolation_end" in context and context["interpolation_end"] and "seed" in config:
            # Start with a completely fresh frame with the desired seed here,
            # this is the condition for a "key latent" with a preceding interpolation
            self.index = 0
            # Make sure we run a txt2img instead of an img2img to get the image the user wants to see at this point
            del context["prev_image"]

        # A new scene just started, initialize if necessary
        if self.scene_init:
            # The new scene contains a specific override seed (i.e. an "init latent"),
            # make sure we start exactly from the one the user specified
            if "seed" in config:
                self.index = 0
        # TODO proper config overlaying
        config_copy.update(
            {
                "W": self.root_config.width,
                "H": self.root_config.height,
                "prompt": prompt,
                # Offset the seed
                "seed": config_copy["seed"] + self.index,
            }
        )
        # Threshold for enabling highres fix
        if self.root_config.width > 576 or self.root_config.height > 576:
            max_comp = max(self.root_config.width, self.root_config.height)
            divisor = max_comp // 512
            aspect_ratio = max(self.root_config.width / self.root_config.height,
                               self.root_config.height / self.root_config.width)
            larger_comp_scaled = (max_comp // divisor)
            config_copy.update({
                "hires_fix_enabled": "true",
                "H_init": 0,
                "W_init": 0,
                "hires_denoising_strength": 0.9
            })
        else:
            config_copy.update({
                "hires_fix_enabled": "false",
                "H_init": 0,
                "W_init": 0,
                "hires_denoising_strength": 0.9
            })
        if self.index % (config_copy["turbo_steps"] + 1) != 0:
            # Turbo step, override steps params
            config_copy.update({"steps": config_copy.get("turbo_sampling_steps")})

        logging.info(f"Config map for api mechanism {config_copy}")
        # TODO: if the scene has an init seed: discard the prev image and run txt2img if interpolation is 0,
        #  or track interpolation progress and run the txt2img with the start seed at the end of the interpolation
        if "prev_image" not in context:
            img = self._txt2img(config_copy)
        else:
            image_array = context["prev_image"]
            # Contrast adjust test
            # im_mean = np.mean(noised_sample)
            # noised_sample = (noised_sample - im_mean) * 0.9 + im_mean
            img = self._img2img(config_copy, Image.fromarray(image_array))
        self.index = self.index + 1
        # If this was an initialization frame, the next one must not be anymore
        self.scene_init = False
        return img, {
            "prev_frame": np.array(img).astype(np.uint8)
        }

    def _txt2img(self, config_param: dict):
        body = TXT2IMG.format(**config_param)
        res = requests.post(f"{self.host}/api/predict/",
                            data=body)
        if res.status_code == 200:
            json = res.json()
            # We just assume the expected format for now. Errors in the format received from the API lead to crashes.
            # Remove prefix: data:image/png;base64,
            image = json['data'][0][0]['name']
            return Image.open(image)
        else:
            logging.error(f"API request failed, response code {res.status_code}, response body: {res.raw}")
            logging.error(f"Original request body: {body}")
            raise RuntimeError("Unexpected non-200 exit code")

    def _img2img(self, config_param: dict, img: Image):
        # Encode image
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        # clip off the "byte" indicators in the result string
        img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
        body = IMG2IMG.format(**config_param,
                              pngbase64=img_str)
        res = requests.post(f"{self.host}/api/predict/",
                            data=body)
        if res.status_code == 200:
            json = res.json()
            # We just assume the expected format for now. Errors in the format received from the API lead to crashes.
            image = json['data'][0][0]['name']
            return Image.open(image)
        else:
            logging.error(f"API request failed, response code {res.status_code}, response body: {res.json()}")
            logging.error(f"Original request body: {body}")
            raise RuntimeError("Unexpected non-200 exit code")

    def set_interpolation_state(self, interpolation_frames: typing.List[str], prev_prompt: str = None):
        self.anim_wrapper.set_interpolation_state(interpolation_frames, prev_prompt)

    def reset_scene_state(self):
        self.anim_wrapper.reset_scene_state()
        self.scene_init = True

    def destroy(self):
        super().destroy()

    @staticmethod
    def name():
        return "api"
