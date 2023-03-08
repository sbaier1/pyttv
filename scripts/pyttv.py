import sys, os

basedirs = [os.getcwd()]
if 'google.colab' in sys.modules:
    basedirs.append(
        '/content/gdrive/MyDrive/sd/stable-diffusion-webui')  # hardcode as TheLastBen's colab seems to be the primal source

for basedir in basedirs:
    pyttv_paths_to_ensure = [basedir + '/extensions/pyttv-for-automatic1111-webui/scripts',
                             basedir + '/extensions/sd-webui-controlnet', basedir + '/extensions/pyttv/scripts',
                             basedir + '/scripts/pyttv_helpers/src',
                             basedir + '/extensions/pyttv/scripts/pyttv_helpers/src',
                             basedir + '/extensions/pyttv-for-automatic1111-webui/scripts/pyttv_helpers/src', basedir]

    for pyttv_scripts_path_fix in pyttv_paths_to_ensure:
        if not pyttv_scripts_path_fix in sys.path:
            sys.path.extend([pyttv_scripts_path_fix])

import modules.scripts as wscripts
from modules import script_callbacks
import gradio as gr
import json

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from PIL import Image
# from pyttv_helpers.video_audio_utilities import ffmpeg_stitch_video, make_gifski_gif
# from pyttv_helpers.upscaling import make_upscale_v2
import gc
import torch
from webui import wrap_gradio_gpu_call
import modules.shared as shared
from modules.ui import create_output_panel, plaintext_to_html, wrap_gradio_call
from types import SimpleNamespace
import typing

from PIL.Image import Image
from omegaconf import DictConfig

from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig

from t2v.mechanism.mechanism import Mechanism


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as pyttv_interface:
        components = {}
        dummy_component = gr.Label(visible=False)
        with gr.Row(elem_id='pyttv_progress_row').style(equal_height=False, variant='compact'):
            scenario = gr.Textbox("# paste your pyttv configuration to execute here", elem_id='scenario',
                                  label="Configuration", max_lines=2000)
            btn = gr.Button("Generate")
            components['btn'] = btn
            btn.click(run_scenario, input=scenario)

    return [(pyttv_interface, "pyttv", "pyttv_interface")]


script_callbacks.on_ui_tabs(on_ui_tabs)


class AutoUiMechanism(Mechanism):
    def __init__(self, config: DictConfig, root_config: RootConfig, func_util: FuncUtil):
        super().__init__(config, root_config, func_util)

    def generate(self, config: DictConfig, context, prompt: str, t):
        super().generate(config, context, prompt, t)

    def destroy(self):
        super().destroy()

    def set_interpolation_state(self, interpolation_frames: typing.List[str], prev_prompt: str = None):
        super().set_interpolation_state(interpolation_frames, prev_prompt)

    def reset_scene_state(self):
        super().reset_scene_state()

    def skip_frame(self, config):
        super().skip_frame(config)

    @staticmethod
    def blend_frames(image1: Image, image2: Image, factor: float):
        return super().blend_frames(image1, image2, factor)

    def simulate_step(self, config, t) -> dict:
        return super().simulate_step(config, t)

    @staticmethod
    def name():
        super().name()
