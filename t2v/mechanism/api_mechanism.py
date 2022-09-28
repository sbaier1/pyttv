import base64
import io
import logging

import numpy as np
from PIL import Image
from omegaconf import DictConfig

from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig

from t2v.mechanism.mechanism import Mechanism

import requests

# Generated with revision e22ea454a273b3a8a807a5acb2e6f0d0d41c9aa7,
# diff the template with a captured query of a more recent version to update these when necessary
from t2v.mechanism.turbo_stablediff_functions import add_noise, sample_from_cv2, sample_to_cv2, maintain_colors

TXT2IMG = """
{{
"fn_index": 12,
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
    null,
    "",
    false,
    "Seed",
    "",
    "Steps",
    "",
    true,
    null,
    "",
    ""
  ],
  "session_hash": "djrqd1giwif"
}}
"""

IMG2IMG = """
{{
"fn_index": 30,
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
        null,
        "",
        false,
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
        self.host = self.config.get("host")
        self.index = 0
        self.color_match_sample = None

    def generate(self, config: DictConfig, context, prompt: str, t):
        # TODO: template out the queries, run txt2img, decode the image
        # TODO: on subsequent steps, run img2img, encode the previous frame as base64 and use as input
        # TODO: warping code from other mechanism

        # TODO move into base class method
        # TODO not sure if this works as intended for nested dicts
        # config overlaying
        if config is not None:
            config_param = self.config.copy()
            config_param.update(config)
        else:
            config_param = self.config
        if self.index % (self.config["turbo_steps"] + 1) != 0:
            # Turbo step, override steps params
            config_param.get("txt2img_params").update(steps=config_param.get("turbo_sampling_steps"))
        if "prev_frame" not in context:
            img = self._txt2img(prompt, config_param)
        else:
            prev_frame = context["prev_frame"]
            image_array = np.array(prev_frame).astype(np.uint8)
            warped_frame = self.animator.apply(image_array, prompt,
                                               self.config.get("animation_parameters"), t)
            if self.color_match_sample is not None:
                warped_frame = maintain_colors(warped_frame, self.color_match_sample, 'Match Frame 0 LAB')
            else:
                self.color_match_sample = image_array
            noised_sample = add_noise(sample_from_cv2(warped_frame),
                                      self.func_util.parametric_eval(config_param.get("noise_schedule"), t))
            noised_sample = sample_to_cv2(noised_sample)
            img = self._img2img(prompt, config_param, Image.fromarray(noised_sample))
        self.index = self.index + 1
        return img, {
            "prev_frame": img
        }

    def _txt2img(self, prompt: str, config_param: DictConfig):
        body = TXT2IMG.format(prompt=prompt, **config_param.get("txt2img_params"),
                              seed=config_param.get("seed") + self.index, W=self.root_config.width,
                              H=self.root_config.height)
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

    def _img2img(self, prompt: str, config_param: DictConfig, img: Image):
        # Encode image
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        # clip off the "byte" indicators in the result string
        img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
        body = IMG2IMG.format(prompt=prompt, **config_param.get("img2img_params"), pngbase64=img_str,
                              seed=config_param.get("seed") + self.index, W=self.root_config.width,
                              H=self.root_config.height)
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

    def destroy(self):
        super().destroy()

    @staticmethod
    def name():
        return "api"
