import logging

import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from t2v.mechanism.mechanism import Mechanism
from t2v.mechanism.t2i_3d_anim_wrapper import T2IAnimatedWrapper
from t2v.mechanism.turbo_stablediff_functions import DeforumArgs, TurboStableDiffUtils, sample_to_cv2, sample_from_cv2


class TurboStableDiff(Mechanism):
    def __init__(self, config, root_config, func_util):
        super().__init__(config, root_config, func_util)
        self.func_util = func_util
        func_util.add_callback("isTurboStep", self.is_turbo_step)
        model_path = config["model_path"]
        logging.info(f"Initializing StableDiffusion model, this may take a while...")
        self.config = config
        self.root_config = root_config
        self.model = load_model_from_config(config["model_config_path"], model_path, device=root_config.torch_device)
        logging.info("StableDiffusion model loaded")
        self.device = torch.device(root_config.torch_device)
        self.utils = TurboStableDiffUtils(model=self.model, device=self.device,
                                          args=DeforumArgs(dict(config), root_config))
        # Counter how many frames this instance has generated
        self.index = 0
        self.anim_wrapper = T2IAnimatedWrapper(config, root_config, func_util, self.actual_generate)

        # TODO resume last frame
        # path = os.path.join(args.outdir,f"{args.timestring}_{last_frame:05}.png")
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.color_match_sample = None

    def is_turbo_step(self):
        return self.index % (self.config["turbo_steps"] + 1) == 0

    def generate(self, config, context, prompt, t):
        return self.anim_wrapper.generate(config, context, prompt, t)

    def actual_generate(self, config, context, prompt, t):
        # TODO: seed method should be configurable
        seed_everything(self.config["seed"] + self.index)
        # Overlay base config with overrides from current scene
        if config is not None:
            config_new = self.config.copy()
            config_new.update(config)
            config = config_new
        else:
            config = self.config
        # Add "overrides"
        args = {
            "prompt": prompt,
            "W": self.root_config.width,
            "H": self.root_config.height,
            "f": 8,
            "C": 4,
            "n_samples": 1
        }
        args.update(self.config)
        args.update(config)
        strength_evaluated = self.func_util.parametric_eval(config.get("strength_schedule"), t)

        if "prev_image" in context or len(self.interpolation_frames) > 0:
            prev_image = sample_from_cv2(context["prev_image"])

            if args["half_precision"]:
                prev_image = prev_image.half().to(self.device)
            else:
                prev_image = prev_image.to(self.device)
            args["init_sample"] = prev_image
            args["use_init"] = True
            args["strength"] = strength_evaluated
            # TODO: can dynamic thresholding be useful? more testing needed. static seed?
            # args["dynamic_threshold"] = 1
        if self.index % (self.config["turbo_steps"] + 1) == 0:
            # Standard step
            logging.info(f"Standard step at index {self.index}, steps: {args['steps']}")
            samples, image = self.utils.generate(args, return_sample=True)
        else:
            # Turbo step
            logging.info(f"Turbo step at index {self.index}, steps: {args['turbo_sampling_steps']}")
            args["steps"] = args["turbo_sampling_steps"]
            samples, image = self.utils.generate(args, return_sample=True)

        if self.color_match_sample is None:
            self.color_match_sample = sample_to_cv2(samples)

        self.index = self.index + 1

        return image, {
            "prev_samples": samples,
        }

    def destroy(self):
        super().destroy()

    def reset_scene_state(self):
        self.color_match_sample = None

    @staticmethod
    def name():
        return "turbo-stablediff"


def load_model_from_config(config_path, ckpt, verbose=False, device='cuda', half_precision=True):
    config = OmegaConf.load(config_path)
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
