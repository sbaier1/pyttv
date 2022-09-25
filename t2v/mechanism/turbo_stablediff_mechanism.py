import logging
import os
import sys

import torch
from omegaconf import OmegaConf

from t2v.animation.animator_3d import Animator3D
from t2v.mechanism.mechanism import Mechanism
from ldm.util import instantiate_from_config
from t2v.mechanism.turbo_stablediff_functions import DeforumArgs, TurboStableDiffUtils


class TurboStableDiff(Mechanism):
    def __init__(self, config, root_config):
        super().__init__(config, root_config)
        model_path = config["model_path"]
        model_config_path = config["model_config_path"]
        logging.info(f"Initializing StableDiffusion model, this may take a while...")
        local_config = OmegaConf.load(model_config_path)
        self.config = config
        self.root_config = root_config
        self.model = load_model_from_config(local_config, model_path, device=root_config.torch_device)
        logging.info("StableDiffusion model loaded")
        self.utils = TurboStableDiffUtils(model=self.model, device=torch.device(root_config.torch_device),
                                          args=DeforumArgs(dict(config), root_config))
        # Counter how many frames this instance has generated
        self.index = 0

        animation_type = config.get("animation")
        if animation_type is not None:
            if animation_type == "3D":
                self.animator = Animator3D(config.get("animation_parameters"), root_config)
            elif animation_type == "2D":
                # TODO 2D anim
                self.animator = None

    def generate(self, config, context, prompt, t):
        args = dict(self.config)
        # Add "overrides"
        args.update(config)
        args["prompt"] = prompt
        args["W"] = self.root_config.width
        args["H"] = self.root_config.height
        args["f"] = 8
        args["C"] = 4
        args["n_samples"] = 1
        if "warped_frame" in context:
            # Use prev warped frame if exists
            args["init_image"] = context["warped_frame"]
        samples, image = self.utils.generate(args)
        self.index = self.index + 1
        warped_frame = self.animator.apply(image, prompt, config.get("animation_parameters"), t)

        # apply color matching
        # if anim_args.color_coherence != 'None':
        #    if color_match_sample is None:
        #        color_match_sample = prev_img.copy()
        #    else:
        #        prev_img = self.maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

        # apply scaling
        # contrast_sample = prev_img * contrast
        # apply frame noising
        # noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

        return image, {
            "samples": samples,
            "warped_frame": warped_frame,
        }

    def destroy(self):
        super().destroy()

    @staticmethod
    def name():
        return "turbo-stablediff"


def load_model_from_config(config, ckpt, verbose=False, device='cuda', half_precision=True):
    map_location = device
    if device == 'mps':
        # We load to RAM for mps but use mps for inference
        map_location = "cpu"
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=map_location)
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    if half_precision:
        model = model.half().to(device)
    else:
        model = model.to(device)
    model.eval()
    return model
