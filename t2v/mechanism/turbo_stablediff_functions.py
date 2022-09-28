import json
import os
import pathlib
import random
import subprocess
import time
from contextlib import nullcontext

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange, repeat
from k_diffusion.external import CompVisDenoiser
from pytorch_lightning import seed_everything
from skimage.exposure import match_histograms
from torch import autocast

from t2v.config.root import RootConfig
from t2v.mechanism.helpers_stablediff.depth import DepthModel
from t2v.mechanism.helpers_stablediff.k_samplers import sampler_fn
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def DeforumArgs(params: dict, root_config: RootConfig):
    W = int(root_config.width)
    H = int(root_config.height)
    # resize to integer multiple of 64
    W, H = map(lambda x: x - x % 64, (W, H))

    seed = int(params['seed'])
    sampler = params['sampler']
    steps = int(params['steps'])
    scale = float(params['scale'])
    ddim_eta = float(params['ddim_eta'])
    dynamic_threshold = params['dynamic_threshold']
    static_threshold = params['static_threshold']

    # TODO dict map
    save_samples = True
    save_settings = True
    n_batch = 1
    strength_0_no_init = True  # Set the strength to 0 automatically when no init image is used

    # This stuff should probably go into main ttv utils
    # batch_name = params['batch_name']
    filename_format = "{index}.png"
    half_precision = bool(params['half_precision'])

    seed_behavior = "iter"

    use_init = bool(params['use_init'])
    strength = float(params['init_strength'])
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"
    # Whiter areas of the mask are areas that change more
    use_mask = bool(params['init_use_mask'])
    use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"

    invert_mask = False  # @param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  # @param {type:"number"}
    mask_contrast_adjust = 1.0  # @param {type:"number"}

    n_samples = 1  # doesnt do anything
    precision = 'autocast'
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None
    return locals()


def next_seed(args: DeforumArgs):
    if args.seed_behavior == 'iter':
        args.seed += 1
    elif args.seed_behavior == 'fixed':
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2 ** 32 - 1)
    return args.seed


def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear'):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def parse_key_frames(string, prompt_parser=None):
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)


def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample - (noise_amt/2) + torch.randn(sample.shape, device=sample.device) * noise_amt


def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path, time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path


def load_img(path, shape, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.

    return image, mask_image


def sanitize(prompt):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tmp = ''.join(filter(whitelist.__contains__, prompt))
    return tmp.replace(' ', '_')


# Actually stateful utils that require model and device
class TurboStableDiffUtils:
    def __init__(self, model, device, args: DeforumArgs):
        self.model = model
        self.device = device
        self.args = args

    def load_mask_latent(self, mask_input, shape):
        # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
        # shape (list-like len(4)): shape of the image to match, usually latent_image.shape

        if isinstance(mask_input, str):  # mask input is probably a file name
            if mask_input.startswith('http://') or mask_input.startswith('https://'):
                mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
            else:
                mask_image = Image.open(mask_input).convert('RGBA')
        elif isinstance(mask_input, Image.Image):
            mask_image = mask_input
        else:
            raise Exception("mask_input must be a PIL image or a file name")

        mask_w_h = (shape[-1], shape[-2])
        mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
        mask = mask.convert("L")
        return mask

    def prepare_mask(self, mask_input, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0):
        # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
        # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
        # mask_brightness_adjust (non-negative float): amount to adjust brightness of the iamge,
        #     0 is black, 1 is no adjustment, >1 is brighter
        # mask_contrast_adjust (non-negative float): amount to adjust contrast of the image,
        #     0 is a flat grey image, 1 is no adjustment, >1 is more contrast

        mask = self.load_mask_latent(mask_input, mask_shape)

        # Mask brightness/contrast adjustments
        if mask_brightness_adjust != 1:
            mask = TF.adjust_brightness(mask, mask_brightness_adjust)
        if mask_contrast_adjust != 1:
            mask = TF.adjust_contrast(mask, mask_contrast_adjust)

        # Mask image to array
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)

        if self.args.invert_mask:
            mask = ((mask - 0.5) * -1) + 0.5

        mask = np.clip(mask, 0, 1)
        return mask

    def make_callback(self, sampler_name, dynamic_threshold=None, static_threshold=None, mask=None, init_latent=None,
                      sigmas=None,
                      sampler=None, masked_noise_modifier=1.0):
        # Creates the callback function to be passed into the samplers
        # The callback function is applied to the image at each step
        def dynamic_thresholding_(img, threshold):
            # Dynamic thresholding from Imagen paper (May 2022)
            s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1, img.ndim)))
            s = np.max(np.append(s, 1.0))
            torch.clamp_(img, -1 * s, s)
            torch.FloatTensor.div_(img, s)

        # Callback for samplers in the k-diffusion repo, called thus:
        #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        def k_callback_(args_dict):
            if dynamic_threshold is not None:
                dynamic_thresholding_(args_dict['x'], dynamic_threshold)
            if static_threshold is not None:
                torch.clamp_(args_dict['x'], -1 * static_threshold, static_threshold)
            if mask is not None:
                init_noise = init_latent + noise * args_dict['sigma']
                is_masked = torch.logical_and(mask >= mask_schedule[args_dict['i']], mask != 0)
                new_img = init_noise * torch.where(is_masked, 1, 0) + args_dict['x'] * torch.where(is_masked, 0, 1)
                args_dict['x'].copy_(new_img)

        # Function that is called on the image (img) and step (i) at each step
        def img_callback_(img, i):
            # Thresholding functions
            if dynamic_threshold is not None:
                dynamic_thresholding_(img, dynamic_threshold)
            if static_threshold is not None:
                torch.clamp_(img, -1 * static_threshold, static_threshold)
            if mask is not None:
                i_inv = len(sigmas) - i - 1
                init_noise = sampler.stochastic_encode(init_latent, torch.tensor([i_inv] * batch_size).to(self.device),
                                                       noise=noise)
                is_masked = torch.logical_and(mask >= mask_schedule[i], mask != 0)
                new_img = init_noise * torch.where(is_masked, 1, 0) + img * torch.where(is_masked, 0, 1)
                img.copy_(new_img)

        if init_latent is not None:
            noise = torch.randn_like(init_latent, device=self.device) * masked_noise_modifier
        if sigmas is not None and len(sigmas) > 0:
            mask_schedule, _ = torch.sort(sigmas / torch.max(sigmas))
        elif len(sigmas) == 0:
            mask = None  # no mask needed if no steps (usually happens because strength==1.0)
        if sampler_name in ["plms", "ddim"]:
            # Callback function formated for compvis latent diffusion samplers
            if mask is not None:
                assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"
                batch_size = init_latent.shape[0]

            callback = img_callback_
        else:
            # Default callback function uses k-diffusion sampler variables
            callback = k_callback_

        return callback

    def generate(self, args, return_latent=False, return_sample=False, return_c=False):

        # Don't we need k samplers here too?
        sampler = PLMSSampler(self.model) if args["sampler"] == 'plms' else DDIMSampler(self.model)
        model_wrap = CompVisDenoiser(self.model)
        batch_size = args["n_samples"]
        prompt = args["prompt"]
        assert prompt is not None
        data = [batch_size * [prompt]]
        precision_scope = autocast if args["precision"] == "autocast" else nullcontext

        init_latent = None
        mask_image = None
        init_image = None
        if "init_latent" in args:
            init_latent = args["init_latent"]
        elif "init_sample" in args is not None:
            with precision_scope("cuda"):
                init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(args["init_sample"]))
        elif args["use_init"] and args["init_image"] is not None and args["init_image"] != '':
            init_image, mask_image = load_img(args["init_image"],
                                              shape=(args["W"], args["H"]),
                                              use_alpha_as_mask=args["use_alpha_as_mask"] if hasattr(args,
                                                                                                     "use_alpha_as_mask") else False)
            init_image = init_image.to(self.device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            with precision_scope("cuda"):
                init_latent = self.model.get_first_stage_encoding(
                    self.model.encode_first_stage(init_image))  # move to latent space

        if not args["use_init"] and args["strength"] > 0:
            print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
            print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
            args["strength"] = 0

        # Mask functions
        if args["use_mask"]:
            assert args[
                       "mask_file"] is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
            assert args["use_init"], "use_mask==True: use_init is required for a mask"
            assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"

            mask = self.prepare_mask(args["mask_file"] if mask_image is None else mask_image,
                                     init_latent.shape,
                                     args["mask_contrast_adjust"],
                                     args["mask_brightness_adjust"])

            if (torch.all(mask == 0) or torch.all(mask == 1)) and args["use_alpha_as_mask"]:
                raise Warning(
                    "use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")

            mask = mask.to(self.device)
            mask = repeat(mask, '1 ... -> b ...', b=batch_size)
        else:
            mask = None

        t_enc = int((1.0 - args["strength"]) * args["steps"])

        # Noise schedule for the k-diffusion samplers (used for masking)
        k_sigmas = model_wrap.get_sigmas(args["steps"])
        k_sigmas = k_sigmas[len(k_sigmas) - t_enc - 1:]

        if args["sampler"] in ['plms', 'ddim']:
            sampler.make_schedule(ddim_num_steps=args["steps"], ddim_eta=args["ddim_eta"], ddim_discretize='fill',
                                  verbose=False)

        callback = self.make_callback(sampler_name=args["sampler"],
                                      dynamic_threshold=args["dynamic_threshold"],
                                      static_threshold=args["static_threshold"],
                                      mask=mask,
                                      init_latent=init_latent,
                                      sigmas=k_sigmas,
                                      sampler=sampler)

        results = []
        with torch.no_grad():
            # TODO conditional precision scope
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for prompts in data:
                        uc = None
                        if args["scale"] != 1.0:
                            uc = self.model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)

                        if "init_c" in args:
                            c = args["init_c"]

                        if args["sampler"] in ["klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"]:
                            samples = sampler_fn(
                                c=c,
                                uc=uc,
                                args=args,
                                model_wrap=model_wrap,
                                init_latent=init_latent,
                                t_enc=t_enc,
                                device=self.device,
                                cb=callback)
                        else:
                            # args["sampler"] == 'plms' or args["sampler"] == 'ddim':
                            if init_latent is not None and args["strength"] > 0:
                                z_enc = sampler.stochastic_encode(init_latent,
                                                                  torch.tensor([t_enc] * batch_size).to(self.device))
                            else:
                                z_enc = torch.randn(
                                    [args["n_samples"], args["C"], args["H"] // args["f"], args["W"] // args["f"]],
                                    device=self.device)
                            if args["sampler"] == 'ddim':
                                samples = sampler.decode(z_enc,
                                                         c,
                                                         t_enc,
                                                         unconditional_guidance_scale=args["scale"],
                                                         unconditional_conditioning=uc,
                                                         img_callback=callback)
                            elif args["sampler"] == 'plms':  # no "decode" function in plms, so use "sample"
                                shape = [args["C"], args["H"] // args["f"], args["W"] // args["f"]]
                                samples, _ = sampler.sample(S=args["steps"],
                                                            conditioning=c,
                                                            batch_size=args["n_samples"],
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=args["scale"],
                                                            unconditional_conditioning=uc,
                                                            eta=args["ddim_eta"],
                                                            x_T=z_enc,
                                                            img_callback=callback)
                            else:
                                raise Exception(f"Sampler {args['sampler']} not recognised.")

                        if return_latent:
                            results.append(samples.clone())

                        x_samples = self.model.decode_first_stage(samples)
                        if return_sample:
                            results.append(x_samples.clone())

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if return_c:
                            results.append(c.clone())

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            image = Image.fromarray(x_sample.astype(np.uint8))
                            results.append(image)
        return results

    def DeforumAnimArgs(self, ):
        # @markdown ####**Animation:**
        animation_mode = 'None'  # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
        max_frames = 1000  # @param {type:"number"}
        border = 'wrap'  # @param ['wrap', 'replicate'] {type:'string'}

        # @markdown ####**Motion Parameters:**
        angle = "0:(0)"  # @param {type:"string"}
        zoom = "0:(1.04)"  # @param {type:"string"}
        translation_x = "0:(0)"  # @param {type:"string"}
        translation_y = "0:(0)"  # @param {type:"string"}
        translation_z = "0:(10)"  # @param {type:"string"}
        rotation_3d_x = "0:(0)"  # @param {type:"string"}
        rotation_3d_y = "0:(0)"  # @param {type:"string"}
        rotation_3d_z = "0:(0)"  # @param {type:"string"}
        noise_schedule = "0: (0.02)"  # @param {type:"string"}
        strength_schedule = "0: (0.65)"  # @param {type:"string"}
        contrast_schedule = "0: (1.0)"  # @param {type:"string"}

        # @markdown ####**Coherence:**
        color_coherence = 'Match Frame 0 LAB'  # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
        diffusion_cadence = '1'  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

        # @markdown ####**3D Depth Warping:**
        use_depth_warping = True  # @param {type:"boolean"}
        midas_weight = 0.3  # @param {type:"number"}
        near_plane = 200
        far_plane = 10000
        fov = 40  # @param {type:"number"}
        padding_mode = 'border'  # @param ['border', 'reflection', 'zeros'] {type:'string'}
        sampling_mode = 'bicubic'  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
        save_depth_maps = False  # @param {type:"boolean"}

        # @markdown ####**Video Input:**
        video_init_path = '/content/video_in.mp4'  # @param {type:"string"}
        extract_nth_frame = 1  # @param {type:"number"}

        # @markdown ####**Interpolation:**
        interpolate_key_frames = False  # @param {type:"boolean"}
        interpolate_x_frames = 4  # @param {type:"number"}

        # @markdown ####**Resume Animation:**
        resume_from_timestring = False  # @param {type:"boolean"}
        resume_timestring = "20220829210106"  # @param {type:"string"}

        return locals()

    def render_animation(self, args: DeforumArgs, anim_args):
        # animations use key framed prompts
        args.prompts = animation_prompts

        # expand key frame strings to values
        keys = DeforumAnimKeys(anim_args)

        # expand prompts out to per-frame
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])
        for i, prompt in animation_prompts.items():
            prompt_series[i] = prompt
        prompt_series = prompt_series.ffill().bfill()

        # check for video inits
        using_vid_init = anim_args.animation_mode == 'Video Input'

        # state for interpolating between diffusion steps
        turbo_steps = 1 if using_vid_init else int(anim_args.diffusion_cadence)
        turbo_prev_image, turbo_prev_frame_idx = None, 0
        turbo_next_image, turbo_next_frame_idx = None, 0

        # resume animation
        prev_sample = None
        color_match_sample = None
        if anim_args.resume_from_timestring:
            last_frame = start_frame - 1
            if turbo_steps > 1:
                last_frame -= last_frame % turbo_steps
            path = os.path.join(args.outdir, f"{args.timestring}_{last_frame:05}.png")
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prev_sample = sample_from_cv2(img)
            if anim_args.color_coherence != 'None':
                color_match_sample = img
            if turbo_steps > 1:
                turbo_next_image, turbo_next_frame_idx = sample_to_cv2(prev_sample, type=np.float32), last_frame
                turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
                start_frame = last_frame + turbo_steps

        args.n_samples = 1
        frame_idx = start_frame
        while frame_idx < anim_args.max_frames:
            print(f"Rendering animation frame {frame_idx} of {anim_args.max_frames}")
            noise = keys.noise_schedule_series[frame_idx]
            strength = keys.strength_schedule_series[frame_idx]
            contrast = keys.contrast_schedule_series[frame_idx]
            depth = None

            # emit in-between frames
            if turbo_steps > 1:
                tween_frame_start_idx = max(0, frame_idx - turbo_steps)
                for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                    tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(
                        frame_idx - tween_frame_start_idx)
                    print(f"  creating in between frame {tween_frame_idx} tween:{tween:0.2f}")

                    advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                    advance_next = tween_frame_idx > turbo_next_frame_idx

                    if depth_model.model is not None:
                        assert (turbo_next_image is not None)
                        depth = depth_model.model.predict(turbo_next_image, anim_args)

                    if anim_args.animation_mode == '2D':
                        if advance_prev:
                            turbo_prev_image = anim_frame_warp_2d(turbo_prev_image, self.args, anim_args, keys,
                                                                  tween_frame_idx)
                        if advance_next:
                            turbo_next_image = anim_frame_warp_2d(turbo_next_image, self.args, anim_args, keys,
                                                                  tween_frame_idx)
                    else:  # '3D'
                        if advance_prev:
                            turbo_prev_image = self.anim_frame_warp_3d(turbo_prev_image, depth, anim_args, keys,
                                                                       tween_frame_idx)
                        if advance_next:
                            turbo_next_image = self.anim_frame_warp_3d(turbo_next_image, depth, anim_args, keys,
                                                                       tween_frame_idx)
                    turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                    if turbo_prev_image is not None and tween < 1.0:
                        img = turbo_prev_image * (1.0 - tween) + turbo_next_image * tween
                    else:
                        img = turbo_next_image

                    filename = f"{args.timestring}_{tween_frame_idx:05}.png"
                    cv2.imwrite(os.path.join(args.outdir, filename),
                                cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    if anim_args.save_depth_maps:
                        depth_model.model.save(
                            os.path.join(args.outdir, f"{args.timestring}_depth_{tween_frame_idx:05}.png"),
                            depth)
                if turbo_next_image is not None:
                    prev_sample = sample_from_cv2(turbo_next_image)

            # apply transforms to previous frame
            if prev_sample is not None:
                if anim_args.animation_mode == '2D':
                    prev_img = anim_frame_warp_2d(sample_to_cv2(prev_sample), args, anim_args, keys, frame_idx)
                else:  # '3D'
                    prev_img_cv2 = sample_to_cv2(prev_sample)
                    depth = depth_model.model.predict(prev_img_cv2, anim_args) if depth_model.model else None
                    prev_img = self.anim_frame_warp_3d(prev_img_cv2, depth, anim_args, keys, frame_idx)

                # apply color matching
                if anim_args.color_coherence != 'None':
                    if color_match_sample is None:
                        color_match_sample = prev_img.copy()
                    else:
                        prev_img = self.maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

                # apply scaling
                contrast_sample = prev_img * contrast
                # apply frame noising
                noised_sample = add_noise(sample_from_cv2(contrast_sample), noise).transpose(0, 3, 1, 2)

                # use transformed previous frame as init for current
                args.use_init = True
                if self.args.half_precision:
                    args.init_sample = noised_sample.half().to(self.device)
                else:
                    args.init_sample = noised_sample.to(self.device)
                args.strength = max(0.0, min(1.0, strength))

            # grab prompt for current frame
            args.prompt = prompt_series[frame_idx]
            print(f"{args.prompt} {args.seed}")

            # grab init image for current frame
            if using_vid_init:
                init_frame = os.path.join(args.outdir, 'inputframes', f"{frame_idx + 1:04}.jpg")
                print(f"Using video init frame {init_frame}")
                args.init_image = init_frame

            # sample the diffusion model
            sample, image = self.generate(args, return_latent=False, return_sample=True)
            if not using_vid_init:
                prev_sample = sample

            if turbo_steps > 1:
                turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
                turbo_next_image, turbo_next_frame_idx = sample_to_cv2(sample, type=np.float32), frame_idx
                frame_idx += turbo_steps
            else:
                filename = f"{frame_idx:05}.png"
                image.save(os.path.join(args.outdir, filename))
                if anim_args.save_depth_maps:
                    if depth is None:
                        depth = depth_model.model.predict(sample_to_cv2(sample), anim_args)
                    depth_model.model.save(os.path.join(args.outdir, f"{args.timestring}_depth_{frame_idx:05}.png"),
                                           depth)
                frame_idx += 1

            args.seed = next_seed(args)

    def render_input_video(self, args: DeforumArgs, anim_args):
        # create a folder for the video input frames to live in
        video_in_frame_path = os.path.join(args.outdir, 'inputframes')
        os.makedirs(video_in_frame_path, exist_ok=True)

        # save the video frames from input video
        print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}...")
        try:
            for f in pathlib.Path(video_in_frame_path).glob('*.jpg'):
                f.unlink()
        except:
            pass
        vf = r'select=not(mod(n\,' + str(anim_args.extract_nth_frame) + '))'
        subprocess.run([
            'ffmpeg', '-i', f'{anim_args.video_init_path}',
            '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2',
            '-loglevel', 'error', '-stats',
            os.path.join(video_in_frame_path, '%04d.jpg')
        ], stdout=subprocess.PIPE).stdout.decode('utf-8')

        # determine max frames from length of input frames
        anim_args.max_frames = len([f for f in pathlib.Path(video_in_frame_path).glob('*.jpg')])

        args.use_init = True
        print(
            f"Loading {anim_args.max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}")
        self.render_animation(args, anim_args)

    def render_interpolation(self, args: DeforumArgs, anim_args):
        # animations use key framed prompts
        args.prompts = animation_prompts

        # create output folder for the batch
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Saving animation frames to {args.outdir}")

        # save settings for the batch
        settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
            json.dump(s, f, ensure_ascii=False, indent=4)

        # Interpolation Settings
        args.n_samples = 1
        args.seed_behavior = 'fixed'  # force fix seed at the moment bc only 1 seed is available
        prompts_c_s = []  # cache all the text embeddings

        print(f"Preparing for interpolation of the following...")

        for i, prompt in animation_prompts.items():
            args.prompt = prompt

            # sample the diffusion self.model
            results = self.generate(args, return_c=True)
            c, image = results[0], results[1]
            prompts_c_s.append(c)

            args.seed = next_seed(args)

        print(f"Interpolation start...")

        frame_idx = 0

        if anim_args.interpolate_key_frames:
            for i in range(len(prompts_c_s) - 1):
                dist_frames = list(animation_prompts.items())[i + 1][0] - list(animation_prompts.items())[i][0]
                if dist_frames <= 0:
                    print("key frames duplicated or reversed. interpolation skipped.")
                    return
                else:
                    for j in range(dist_frames):
                        # interpolate the text embedding
                        prompt1_c = prompts_c_s[i]
                        prompt2_c = prompts_c_s[i + 1]
                        args.init_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1 / dist_frames))

                        # sample the diffusion self.model
                        results = self.generate(args)
                        image = results[0]

                        filename = f"{args.timestring}_{frame_idx:05}.png"
                        image.save(os.path.join(args.outdir, filename))
                        frame_idx += 1

                        args.seed = next_seed(args)

        else:
            for i in range(len(prompts_c_s) - 1):
                for j in range(anim_args.interpolate_x_frames + 1):
                    # interpolate the text embedding
                    prompt1_c = prompts_c_s[i]
                    prompt2_c = prompts_c_s[i + 1]
                    args.init_c = prompt1_c.add(
                        prompt2_c.sub(prompt1_c).mul(j * 1 / (anim_args.interpolate_x_frames + 1)))

                    # sample the diffusion self.model
                    results = self.generate(args)
                    image = results[0]

                    filename = f"{args.timestring}_{frame_idx:05}.png"
                    image.save(os.path.join(args.outdir, filename))
                    frame_idx += 1

                    args.seed = next_seed(args)

        # generate the last prompt
        args.init_c = prompts_c_s[-1]
        results = self.generate(args)
        image = results[0]
        filename = f"{args.timestring}_{frame_idx:05}.png"
        image.save(os.path.join(args.outdir, filename))

        args.seed = next_seed(args)

        # clear init_c
        args.init_c = None


class DeforumAnimKeys():
    def __init__(self, anim_args):
        self.angle_series = get_inbetweens(parse_key_frames(anim_args.angle), anim_args.max_frames)
        self.zoom_series = get_inbetweens(parse_key_frames(anim_args.zoom), anim_args.max_frames)
        self.translation_x_series = get_inbetweens(parse_key_frames(anim_args.translation_x),
                                                   anim_args.max_frames)
        self.translation_y_series = get_inbetweens(parse_key_frames(anim_args.translation_y),
                                                   anim_args.max_frames)
        self.translation_z_series = get_inbetweens(parse_key_frames(anim_args.translation_z),
                                                   anim_args.max_frames)
        self.rotation_3d_x_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_x),
                                                   anim_args.max_frames)
        self.rotation_3d_y_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_y),
                                                   anim_args.max_frames)
        self.rotation_3d_z_series = get_inbetweens(parse_key_frames(anim_args.rotation_3d_z),
                                                   anim_args.max_frames)
        self.noise_schedule_series = get_inbetweens(parse_key_frames(anim_args.noise_schedule),
                                                    anim_args.max_frames)
        self.strength_schedule_series = get_inbetweens(parse_key_frames(anim_args.strength_schedule),
                                                       anim_args.max_frames)
        self.contrast_schedule_series = get_inbetweens(parse_key_frames(anim_args.contrast_schedule),
                                                       anim_args.max_frames)


def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:  # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
