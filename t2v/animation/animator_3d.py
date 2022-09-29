import logging
import sys

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig

from t2v.animation.animator import Animator
from t2v.animation.helpers import AnimationUtils
from t2v.config.root import RootConfig
from t2v.mechanism.helpers_stablediff.depth import DepthModel

sys.path.extend([
    'src/AdaBins',
    'src/MiDaS',
    'src/pytorch3d-lite',
])
import py3d_tools as p3d


class Animator3D(Animator):
    def __init__(self, config: DictConfig, root_config: RootConfig, func_util):
        super().__init__(config, root_config)
        self.config = config
        self.root_config = root_config

        self.device = torch.device(root_config.torch_device)
        self.animation_utils = AnimationUtils(self.device)
        self.depth_model = DepthModel(self.device)
        self.depth_model.load_midas(root_config.persistence_dir)
        self.midas_weight = config.get("midas_weight", 1.0)
        if self.midas_weight < 1.0:
            self.depth_model.load_adabins()
        self.func_tool = func_util

    def apply(self, frame, prompt, context, t):
        depth = self.depth_model.predict(frame, self.midas_weight)

        TRANSLATION_SCALE = 1.0 / 200.0  # matches Disco
        translate_xyz = [
            -self.func_tool.parametric_eval(context["translation_x"], t) * TRANSLATION_SCALE,
            self.func_tool.parametric_eval(context["translation_y"], t) * TRANSLATION_SCALE,
            -self.func_tool.parametric_eval(context["translation_z"], t) * TRANSLATION_SCALE
        ]
        rotate_xyz = [
            # FIXME no-op for now, not encoded yet
            0, 0, 0
        ]
        rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=self.device), "XYZ").unsqueeze(0)
        logging.info(f"Applying 3D transform with translate mat {translate_xyz}, rotate mat {rotate_xyz}")
        result = self.transform_image_3d(frame, depth, rot_mat, translate_xyz, context)
        torch.cuda.empty_cache()

        return result

    def destroy(self, config):
        super().destroy(config)
        del self.depth_model

    def transform_image_3d(self, prev_img_cv2, depth_tensor, rot_mat, translate, anim_args):
        # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion
        w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

        aspect_ratio = float(w) / float(h)
        near, far, fov_deg = anim_args["near_plane"], anim_args["far_plane"], anim_args["fov"]
        persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True,
                                                  device=self.device)
        persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat,
                                                  T=torch.tensor([translate]), device=self.device)

        # range of [-1,1] is important to torch grid_sample's padding handling
        y, x = torch.meshgrid(torch.linspace(-1., 1., h, dtype=torch.float32, device=self.device),
                              torch.linspace(-1., 1., w, dtype=torch.float32, device=self.device))
        z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=self.device)
        xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

        xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]
        xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]

        offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
        # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
        identity_2d_batch = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=self.device).unsqueeze(0)
        # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
        coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1, 1, h, w], align_corners=False)
        offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h, w, 2)).unsqueeze(0)

        image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(self.device)
        new_image = torch.nn.functional.grid_sample(
            image_tensor.add(1 / 512 - 0.0001).unsqueeze(0),
            offset_coords_2d,
            mode=anim_args["sampling_mode"],
            padding_mode=anim_args["padding_mode"],
            align_corners=False
        )

        # convert back to cv2 style numpy array
        result = rearrange(
            new_image.squeeze().clamp(0, 255),
            'c h w -> h w c'
        ).cpu().numpy().astype(prev_img_cv2.dtype)
        return result
