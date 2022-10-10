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
# Generated with revision f7c787eb7c295c27439f4fbdf78c26b8389560be,
# diff the template with a captured query of a more recent version to update these when necessary
from t2v.mechanism.turbo_stablediff_functions import add_noise, sample_from_cv2, sample_to_cv2, maintain_colors

TXT2IMG = """
{{
"fn_index": 11,
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
    false,
    false,
    0.7,
    "None",
    false,
    false,
    null,
    "",
    "Seed",
    "",
    "Steps",
    "",
    true,
    false,
    null,
    "",
    ""
  ],
  "session_hash": "djrqd1giwif"
}}
"""

IMG2IMG = """
{{
"fn_index": 29,
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
        "",
        1,
        50,
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
        false,
        false,
        null,
        "",
        "",
        64,
        "None",
        "Seed",
        "",
        "Steps",
        "",
        true,
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

    def is_turbo_step(self, t):
        if self.index % (self.config["turbo_steps"] + 1) == 0:
            return {"is_turbo_step": 0}
        else:
            return {"is_turbo_step": 1}

    def generate(self, config: DictConfig, context, prompt: str, t):
        return self.anim_wrapper.generate(config, context, prompt, t)

    def actual_generate(self, config: DictConfig, context, prompt: str, t):
        # TODO: template out the queries, run txt2img, decode the image
        # TODO: on subsequent steps, run img2img, encode the previous frame as base64 and use as input
        # TODO: warping code from other mechanism

        # if self.index % (self.config["turbo_steps"] + 1) != 0:
        # Turbo step, override steps params
        # TODO: this incorrectly permanently overrides the config
        # config.get("txt2img_params").update(steps=config.get("turbo_sampling_steps"))
        if "prev_image" not in context:
            img = self._txt2img(prompt, config, t)
        else:
            image_array = context["prev_image"]
            # Contrast adjust test
            # im_mean = np.mean(noised_sample)
            # noised_sample = (noised_sample - im_mean) * 0.9 + im_mean
            img = self._img2img(prompt, config, Image.fromarray(image_array), t)
        self.index = self.index + 1
        return img, {
            "prev_frame": np.array(img).astype(np.uint8)
        }

    def _txt2img(self, prompt: str, config_param: DictConfig, t):
        config_param.update(seed=config_param.get("seed") + self.index)
        body = TXT2IMG.format(prompt=prompt, **config_param,
                              W=self.root_config.width,
                              H=self.root_config.height,
                              strength=self.func_util.parametric_eval(config_param.get("strength_schedule"), t))
        res = requests.post(f"{self.host}/api/predict/",
                            data=body)
        if res.status_code == 200:
            json = res.json()
            # We just assume the expected format for now. Errors in the format received from the API lead to crashes.
            # Remove prefix: data:image/png;base64,
            image = base64.b64decode(json['data'][0][0][22:])
            return Image.open(io.BytesIO(image))
        else:
            logging.error(f"API request failed, response code {res.status_code}, response body: {res.raw}")
            logging.error(f"Original request body: {body}")
            raise RuntimeError("Unexpected non-200 exit code")

    def _img2img(self, prompt: str, config_param: DictConfig, img: Image, t):
        # Encode image
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        # clip off the "byte" indicators in the result string
        img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
        config_param.update(seed=config_param.get("seed") + self.index)
        body = IMG2IMG.format(prompt=prompt, **config_param,
                              pngbase64=img_str,
                              W=self.root_config.width,
                              H=self.root_config.height,
                              strength=self.func_util.parametric_eval(config_param.get("strength_schedule"), t))
        res = requests.post(f"{self.host}/api/predict/",
                            data=body)
        if res.status_code == 200:
            json = res.json()
            # We just assume the expected format for now. Errors in the format received from the API lead to crashes.
            return Image.open(io.BytesIO(base64.b64decode(json['data'][0][0][22:])))
        else:
            logging.error(f"API request failed, response code {res.status_code}, response body: {res.json()}")
            logging.error(f"Original request body: {body}")
            raise RuntimeError("Unexpected non-200 exit code")

    def set_interpolation_state(self, interpolation_frames: typing.List[str], prev_prompt: str = None):
        self.anim_wrapper.set_interpolation_state(interpolation_frames, prev_prompt)

    def reset_scene_state(self):
        self.anim_wrapper.reset_scene_state()

    def destroy(self):
        super().destroy()

    @staticmethod
    def name():
        return "api"
