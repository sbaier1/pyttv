import base64
import io
import logging
import typing

import numpy as np
import requests
from PIL import Image
from omegaconf import DictConfig

from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig
from t2v.mechanism.mechanism import Mechanism
from t2v.mechanism.t2i_3d_anim_wrapper import T2IAnimatedWrapper

TXT2IMG = """
{{
  "enable_hr": {hires_fix_enabled},
  "denoising_strength": {hires_denoising_strength},
  "firstphase_width": 0,
  "firstphase_height": 0,
  "prompt": "{prompt}",
  "styles": [
    "string"
  ],
  "seed": {seed},
  "subseed": -1,
  "subseed_strength": 0,
  "seed_resize_from_h": -1,
  "seed_resize_from_w": -1,
  "batch_size": 1,
  "n_iter": 1,
  "steps": {steps},
  "cfg_scale": {scale},
  "width": {W},
  "height": {H},
  "restore_faces": false,
  "tiling": false,
  "negative_prompt": "{negative_prompt}",
  "eta": 0,
  "s_churn": 0,
  "s_tmax": 0,
  "s_tmin": 0,
  "s_noise": 1,
  "override_settings": {{}},
  "sampler_index": "{sampler}"
}}
"""

IMG2IMG = """
{{
  "init_images": [
    "data:image/png;base64,{pngbase64}"
  ],
  "resize_mode": 0,
  "denoising_strength": {strength},
  "mask": null,
  "mask_blur": 4,
  "inpainting_fill": 0,
  "inpaint_full_res": true,
  "inpaint_full_res_padding": 0,
  "inpainting_mask_invert": 0,
  "prompt": "{prompt}",
  "styles": [
    "string"
  ],
  "seed": {seed},
  "subseed": {subseed},
  "subseed_strength": 0.985,
  "seed_resize_from_h": -1,
  "seed_resize_from_w": -1,
  "batch_size": 1,
  "n_iter": 1,
  "steps": {steps},
  "cfg_scale": {scale},
  "width": {W},
  "height": {H},
  "restore_faces": false,
  "tiling": false,
  "negative_prompt": "{negative_prompt}",
  "eta": 0,
  "s_churn": 0,
  "s_tmax": 0,
  "s_tmin": 0,
  "s_noise": 0,
  "override_settings": {{}},
  "sampler_index": "{sampler}",
  "include_init_images": false
}}"""


class ApiMechanism(Mechanism):
    def __init__(self, config: DictConfig, root_config: RootConfig, func_util: FuncUtil):
        super().__init__(config, root_config, func_util)
        self.root_config = root_config
        self.config = config
        self.current_config = dict(config)
        self.func_util = func_util
        func_util.add_callback("isTurboStep", self.is_turbo_step)
        self.host = self.config.get("host")
        self.index = 0
        self.anim_wrapper = T2IAnimatedWrapper(config, root_config, func_util, self.actual_generate, self)
        self.scene_init = True

    def is_turbo_step(self, t):
        return {"is_turbo_step": 0 if self.index % (self.current_config["turbo_steps"] + 1) == 0 else 1,
                "index": self.index,
                "interpolation_ongoing": 1 if self.anim_wrapper.interpolation_ongoing else 0}

    def generate(self, config: DictConfig, context, prompt: str, t):
        return self.anim_wrapper.generate(config, context, prompt, t)

    def skip_frame(self):
        self.index = self.index + 1
        self.anim_wrapper.skip_frame()

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
            config_copy.update(
                {"strength": min(0.95, max(0, 1 - self.func_util.parametric_eval(config.get("strength_schedule"), t)))})
        else:
            # Invert input strength because it works the other way round in this mechanism
            config_copy.update({"strength": min(0.95, max(0, 1 - context['strength']))})
        # handle CFG scale schedule if necessary
        config_copy["scale"] = self.func_util.parametric_eval(config_copy.get("scale"), t)
        self.current_config.update(config_copy)
        # TODO: key latents don't work here atm. img2img is too different from txt2img with scaled latents.
        #   idea: implement img2img module for automatic1111 for slerping between prompts, use that to generate all interpolation frames and write them directly.
        #   idea(easier?): generate the key latent image first, then run img2img between prev prompt with down-sloping denoising (denoise 0 in last step to finish transition)
        # if "interpolation_end" in context and context["interpolation_end"] and "seed" in config:
        # Start with a completely fresh frame with the desired seed here,
        # this is the condition for a "key latent" with a preceding interpolation
        # self.index = 0
        # Make sure we run a txt2img instead of an img2img to get the image the user wants to see at this point
        # del context["prev_image"]

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
                "seed": config_copy["seed"],
                "subseed": config_copy["seed"] + self.index,
            }
        )
        # Threshold for enabling highres fix
        if self.root_config.width > 576 or self.root_config.height > 576:
            config_copy.update({
                "hires_fix_enabled": "true",
                "H_init": 0,
                "W_init": 0,
            })
        else:
            config_copy.update({
                "hires_fix_enabled": "false",
                "H_init": 0,
                "W_init": 0,
            })

        # Set a default if not provided
        if "hires_denoising_strength" not in config_copy:
            config_copy["hires_denoising_strength"] = 0.9
        if "negative_prompt" not in config_copy:
            config_copy["negative_prompt"] = ""
        # else:
        #    template_dict = {'math': math,
        #                     'scene_progress': progress,
        #                     }
        #    # Add function context
        #    template_dict.update(self.func_util.update_math_env(self.t))
        #    evaluated_prompt = Template(scene.prompt).render(
        #        template_dict)
        if self.index % (self.func_util.parametric_eval(config_copy["turbo_steps"], t) + 1) != 0:
            # Turbo step, override steps params
            config_copy.update({"steps": config_copy.get("turbo_sampling_steps")})

        if config_copy["strength"] <= 0 and "prev_image" in context:
            logging.info("Skipping img2img due to strength <= 0")
            return Image.fromarray(context["prev_image"]), {
                "prev_frame": context["prev_image"]
            }

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
        res = requests.post(f"{self.host}/sdapi/v1/txt2img",
                            data=body)
        if res.status_code == 200:
            json = res.json()
            # We just assume the expected format for now. Errors in the format received from the API lead to crashes.
            image = json['images'][0]
            return Image.open(io.BytesIO(base64.b64decode(image)))
        else:
            logging.error(f"API request failed, response code {res.status_code}, response body: {res.raw}")
            logging.error(f"Original request body: {body}")
            raise RuntimeError("Unexpected non-200 exit code")

    def _img2img(self, config_param: dict, img: Image):
        # Encode image
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        if config_param["strength"] <= 0:
            logging.info("Skipping img2img due to strength <= 0")
            return img
        # clip off the "byte" indicators in the result string
        img_str = str(base64.b64encode(buffered.getvalue()))[2:-1]
        body = IMG2IMG.format(**config_param,
                              pngbase64=img_str)
        res = requests.post(f"{self.host}/sdapi/v1/img2img",
                            data=body)
        if res.status_code == 200:
            json = res.json()
            # We just assume the expected format for now. Errors in the format received from the API lead to crashes.
            image = json['images'][0]
            return Image.open(io.BytesIO(base64.b64decode(image)))
        else:
            logging.error(f"API request failed, response code {res.status_code}, response body: {res.text}")
            logging.error(f"Original request body: {body}")
            raise RuntimeError("Unexpected non-200 exit code")

    def set_interpolation_state(self, interpolation_frames: typing.List[str], prev_prompt: str = None):
        self.anim_wrapper.set_interpolation_state(interpolation_frames, prev_prompt)

    def reset_scene_state(self):
        self.anim_wrapper.reset_scene_state()
        self.scene_init = True

    def simulate_step(self, config, t) -> dict:
        # Start with the root (default) config
        config_copy = dict(self.config.copy())
        # Overlay the scene-specific params if necessary
        if config is not None:
            config_copy.update(config)
        initial_dict = {
            "strength": self.func_util.parametric_eval(config_copy["strength_schedule"], t),
        }
        initial_dict.update(self.anim_wrapper.simulate_step(config, t))
        self.index += 1
        return initial_dict

    def destroy(self):
        super().destroy()

    @staticmethod
    def name():
        return "api"
