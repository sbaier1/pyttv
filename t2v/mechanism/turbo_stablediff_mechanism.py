import logging

import torch
from omegaconf import OmegaConf

from t2v.mechanism.mechanism import Mechanism
from ldm.util import instantiate_from_config


class TurboStableDiff(Mechanism):
    def init(self, config):
        model_path = config["model_path"]
        model_config_path = config["model_config_path"]
        logging.info(f"Initializing StableDiffusion model, this may take a while...")
        local_config = OmegaConf.load(model_config_path)
        load_model_from_config(local_config, model_path)
        logging.info("StableDiffusion model loaded")

    def generate(self, config, context, prompt):
        super().generate(config, context, prompt)

    def destroy(self):
        super().destroy()

    @staticmethod
    def name():
        return "turbo-stablediff"


def load_model_from_config(config, ckpt, verbose=False, device='cuda', half_precision=True):
    map_location = "cuda"  # @param ["cpu", "cuda"]
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
