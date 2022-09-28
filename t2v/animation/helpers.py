import math
import sys

import cv2
import numpy as np
import torch
from einops import rearrange


class AnimationUtils:
    def __init__(self, device):
        self.device = device

    def anim_frame_warp_2d(prev_img_cv2, args, anim_args, keys, frame_idx):
        angle = keys.angle_series[frame_idx]
        zoom = keys.zoom_series[frame_idx]
        translation_x = keys.translation_x_series[frame_idx]
        translation_y = keys.translation_y_series[frame_idx]

        center = (args.W // 2, args.H // 2)
        trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
        trans_mat = np.vstack([trans_mat, [0, 0, 1]])
        rot_mat = np.vstack([rot_mat, [0, 0, 1]])
        xform = np.matmul(rot_mat, trans_mat)

        return cv2.warpPerspective(
            prev_img_cv2,
            xform,
            (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
            borderMode=cv2.BORDER_WRAP if anim_args.border == 'wrap' else cv2.BORDER_REPLICATE
        )
