import logging

import numpy as np
import torch
from PIL import Image
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from t2v.mechanism.mechanism import Mechanism
from t2v.mechanism.turbo_stablediff_functions import DeforumArgs, TurboStableDiffUtils, sample_to_cv2, sample_from_cv2, \
    add_noise, maintain_colors


class TurboStableDiff(Mechanism):
    def __init__(self, config, root_config, func_util):
        super().__init__(config, root_config, func_util)
        self.func_util = func_util
        model_path = config["model_path"]
        model_config_path = config["model_config_path"]
        logging.info(f"Initializing StableDiffusion model, this may take a while...")
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

        # TODO resume last frame
        # path = os.path.join(args.outdir,f"{args.timestring}_{last_frame:05}.png")
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.color_match_sample = None

    def generate(self, config, context, prompt, t):
        # TODO: seed method should be configurable
        seed_everything(self.config["seed"] + self.index)
        args = dict(self.config)
        # Overlay base config with overrides from current scene
        if config is not None:
            config_new = self.config.copy()
            config_new.update(config)
            config = config_new
        else:
            config = self.config
        # Add "overrides"
        args.update(config)
        args["prompt"] = prompt
        args["W"] = self.root_config.width
        args["H"] = self.root_config.height
        args["f"] = 8
        args["C"] = 4
        args["n_samples"] = 1
        strength_evaluated = self.func_util.parametric_eval(config.get("strength_schedule"), t)
        if "prev_samples" in context or len(self.interpolation_frames) > 0:
            # Use prev frame if exists
            if "prev_samples" in context:
                prev_samples = context["prev_samples"]
                # TODO: should merge config + self.config as input here so animations can be overriden per-scene
                previous_image = sample_to_cv2(prev_samples)
            else:
                previous_image = None
            previous_image, strength_evaluated = self.interpolate(config, previous_image, strength_evaluated, t)
            warped_frame = self.animator.apply(previous_image, prompt,
                                               self.config.get("animation_parameters"), t)
            # cv2.imwrite(os.path.join(self.root_config.output_path, f"{self.index:05}_warped.png"),
            #            cv2.cvtColor(warped_frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
            # apply color matching
            if self.color_match_sample is not None:
                warped_frame = maintain_colors(warped_frame, self.color_match_sample, 'Match Frame 0 LAB')

            # TODO: parameters for contrast schedule, noise schedule
            # apply scaling
            # contrast_sample = warped_frame * 0.95
            # apply frame noising
            # torch.tensor(warped_frame, dtype=torch.float32)
            noised_sample = add_noise(sample_from_cv2(warped_frame),
                                      self.func_util.parametric_eval(config.get("noise_schedule"), t))

            if args["half_precision"]:
                noised_sample = noised_sample.half().to(self.device)
            else:
                noised_sample = noised_sample.to(self.device)
            args["init_sample"] = noised_sample
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
            factor = None
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
                    self.blend_frames(Image.open(interpolation_frame), Image.fromarray(previous_image), factor))
            else:
                previous_image = np.asarray(Image.open(interpolation_frame))
            # modulate the denoising strength while the interpolation is ongoing to retain more of the interpolation frames
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

    def destroy(self):
        super().destroy()

    def reset_scene_state(self):
        self.color_match_sample = None

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
