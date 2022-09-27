import logging
import os
import sys

import cv2
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from t2v.animation.animator_3d import Animator3D
from t2v.mechanism.mechanism import Mechanism
from ldm.util import instantiate_from_config
from t2v.mechanism.turbo_stablediff_functions import DeforumArgs, TurboStableDiffUtils, sample_to_cv2, sample_from_cv2, \
    add_noise


class TurboStableDiff(Mechanism):
    def __init__(self, config, root_config):
        super().__init__(config, root_config)
        model_path = config["model_path"]
        model_config_path = config["model_config_path"]
        logging.info(f"Initializing StableDiffusion model, this may ta  ke a while...")
        local_config = OmegaConf.load(model_config_path)
        self.config = config
        self.root_config = root_config
        self.model = load_model_from_config(local_config, model_path, device=root_config.torch_device)
        logging.info("StableDiffusion model loaded")
        self.device = torch.device(root_config.torch_device)
        self.utils = TurboStableDiffUtils(model=self.model, device=self.device,
                                          args=DeforumArgs(dict(config), root_config))
        # Counter how many frames this instance has generated
        self.index = 0

        # TODO resume initial frame
        # path = os.path.join(args.outdir,f"{args.timestring}_{last_frame:05}.png")
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.color_match_sample = None
        animation_type = config.get("animation")
        if animation_type is not None:
            if animation_type == "3D":
                self.animator = Animator3D(config.get("animation_parameters"), root_config)
            elif animation_type == "2D":
                # TODO 2D anim
                self.animator = None

    def generate(self, config, context, prompt, t):
        # TODO: seed method should be configurable
        seed_everything(self.config["seed"]+self.index)
        args = dict(self.config)
        # Add "overrides"
        args.update(config)
        args["prompt"] = prompt
        args["W"] = self.root_config.width
        args["H"] = self.root_config.height
        args["f"] = 8
        args["C"] = 4
        args["n_samples"] = 1
        if "prev_samples" in context:
            # Use prev frame if exists
            prev_samples = context["prev_samples"]
            # TODO: should merge config + self.config as input here so animations can be overriden per-scene
            warped_frame = self.animator.apply(sample_to_cv2(prev_samples), prompt, self.config.get("animation_parameters"), t)

            # cv2.imwrite(os.path.join(self.root_config.output_path, f"{self.index:05}_warped.png"),
            #            cv2.cvtColor(warped_frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
            # apply color matching
            if self.color_match_sample is not None:
                warped_frame = self.utils.maintain_colors(warped_frame, self.color_match_sample, 'Match Frame 0 LAB')

            # TODO: parameters for contrast schedule, noise schedule
            # apply scaling
            # contrast_sample = warped_frame * 0.95
            # apply frame noising
            #torch.tensor(warped_frame, dtype=torch.float32)
            noised_sample = add_noise(sample_from_cv2(warped_frame), 0.04)

            if args["half_precision"]:
                noised_sample = noised_sample.half().to(self.device)
            else:
                noised_sample = noised_sample.to(self.device)
            args["init_sample"] = noised_sample
            args["use_init"] = True
            # TODO: config param
            args["strength"] = 0.65
        if self.index % (self.config["turbo_steps"] + 1) == 0:
            # Standard step
            logging.info(f"Standard step at index {self.index}, steps: {args['steps']}")
            samples, image = self.utils.generate(args, return_sample=True)
        else:
            # Turbo step
            logging.info(f"Turbo step at index {self.index}, steps: {args['steps']}")
            args["steps"] = args["turbo_sampling_steps"]
            samples, image = self.utils.generate(args, return_sample=True)

        if self.index == 0:
            self.color_match_sample = sample_to_cv2(samples)

        self.index = self.index + 1

        return image, {
            "prev_samples": samples,
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
