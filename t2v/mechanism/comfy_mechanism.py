import io
import json
import logging
import typing
import uuid

import numpy as np
from PIL import Image
from omegaconf import DictConfig

from t2v.animation.func_tools import FuncUtil
from t2v.config.root import RootConfig
from t2v.mechanism.comfy_client import ComfyClient
from t2v.mechanism.mechanism import Mechanism
from t2v.mechanism.t2i_3d_anim_wrapper import T2IAnimatedWrapper


class ComfyMechanism(Mechanism):
    def __init__(self, config: DictConfig, root_config: RootConfig, func_util: FuncUtil):
        super().__init__(config, root_config, func_util)
        self.root_config = root_config
        self.config = config
        self.current_config = dict(config)
        self.func_util = func_util
        
        self.host = self.config.get("host", "127.0.0.1:8188")
        self.client = ComfyClient(self.host)
        
        # Load workflow template (Default / Img2Img)
        with open(self.config.get("workflow_path"), "r") as f:
            self.workflow_template = json.load(f)

        # Load T2I workflow template (Optional)
        self.workflow_template_t2i = None
        if self.config.get("workflow_path_t2i"):
            with open(self.config.get("workflow_path_t2i"), "r") as f:
                self.workflow_template_t2i = json.load(f)

        self.index = 0
        self.anim_wrapper = T2IAnimatedWrapper(config, root_config, func_util, self.actual_generate, self)
        self.scene_init = True
        
        # Mappings: config key -> "node_id.input_key"
        self.input_mappings = self.config.get("input_mappings", {})
        self.input_mappings_t2i = self.config.get("input_mappings_t2i", self.input_mappings)

    def generate(self, config: DictConfig, context, prompt: str, t):
        # Start with the root (default) config
        config_copy = dict(self.config.copy())
        # Overlay the scene-specific params if necessary
        if config is not None:
            config_copy.update(config)
        self.current_config = config_copy
        return self.anim_wrapper.generate(config_copy, context, prompt, t)

    def skip_frame(self, config):
        self.index = self.index + 1
        self.anim_wrapper.skip_frame(config)

    def actual_generate(self, config: DictConfig, context, prompt: str, t):
        is_t2i = "prev_image" not in context
        
        if is_t2i and self.workflow_template_t2i:
            workflow = dict(self.workflow_template_t2i)
            mappings = self.input_mappings_t2i
            logging.info("Using T2I workflow")
        else:
            workflow = dict(self.workflow_template)
            mappings = self.input_mappings
            logging.info("Using default/I2I workflow")

        # 1. Update workflow based on mappings
        self._update_node(workflow, mappings, "prompt", prompt)
        self._update_node(workflow, mappings, "negative_prompt", config.get("negative_prompt", ""))
        
        # Seeds
        seed = config.get("seed", 0) + self.index
        self._update_node(workflow, mappings, "seed", seed)

        # Handle 'init_image' (prev_image)
        if not is_t2i:
            image_array = context["prev_image"]
            img = Image.fromarray(image_array)
            
            # Upload image
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_data = buffered.getvalue()
            filename = f"t2v_init_{uuid.uuid4()}.png"
            
            self.client.upload_image(image_data, filename, subfolder="t2v_inputs")
            
            # Update Load Image node
            self._update_node(workflow, mappings, "init_image", filename)
        
        # Evaluate and map other arbitrary variables
        # Iterate over input_mappings from config
        for key, target in mappings.items():
            if key in ["prompt", "negative_prompt", "seed", "init_image"]:
                continue # Handled above or special
            
            # value can come from context (if calculated by anim wrapper) or config (needs evaluation)
            value = None
            if key in context:
                value = context[key]
            elif key in config:
                raw_val = config[key]
                # If it's a string, try to evaluate it as a schedule/param
                if isinstance(raw_val, str) and "(" in raw_val:
                     value = self.func_util.parametric_eval(raw_val, t)
                else:
                    value = raw_val
            
            if value is not None:
                self._update_node(workflow, mappings, key, value)

        # 2. Queue Prompt
        logging.info(f"Queuing workflow for frame {self.index}")
        response = self.client.queue_prompt(workflow)
        if 'prompt_id' not in response:
            raise RuntimeError(f"ComfyUI Error: {response}")
        prompt_id = response['prompt_id']
        
        # 3. Wait for result
        self.client.wait_for_completion(prompt_id)
        
        # 4. Get Result
        images_data = self.client.get_images(prompt_id)
        if not images_data:
            raise RuntimeError("No images returned from ComfyUI")
            
        # Assume the first image is the result
        result_img = Image.open(io.BytesIO(images_data[0]))
        
        self.index += 1
        self.scene_init = False
        
        return result_img, {
            "prev_frame": np.array(result_img).astype(np.uint8)
        }

    def _update_node(self, workflow, mappings, mapping_key, value):
        """
        Updates a node in the workflow based on the mapping key.
        mapping_key refers to a key in 'mappings', e.g. "prompt".
        """
        target = mappings.get(mapping_key)
        if not target:
            # logging.warning(f"No mapping found for {mapping_key}")
            return

        try:
            node_id, input_name = target.split(".")
            if node_id in workflow:
                if "inputs" in workflow[node_id]:
                    workflow[node_id]["inputs"][input_name] = value
                else:
                    workflow[node_id]["inputs"] = {input_name: value}
            else:
                logging.warning(f"Node {node_id} not found in workflow")
        except ValueError:
            logging.error(f"Invalid mapping format for {mapping_key}: {target}. Expected 'node_id.input_name'")

    def set_interpolation_state(self, interpolation_frames: typing.List[str], prev_prompt: str = None):
        self.anim_wrapper.set_interpolation_state(interpolation_frames, prev_prompt)

    def reset_scene_state(self):
        self.anim_wrapper.reset_scene_state()
        self.scene_init = True
        
    def simulate_step(self, config, t) -> dict:
        self.index = int(t * self.root_config.frames_per_second)
         # Start with the root (default) config
        config_copy = dict(self.config.copy())
        # Overlay the scene-specific params if necessary
        if config is not None:
            config_copy.update(config)
        
        initial_dict = {
            # Add simulations for comfy params if needed?
        }
        initial_dict.update(self.anim_wrapper.simulate_step(config, t))
        return initial_dict

    def destroy(self):
        super().destroy()

    @staticmethod
    def name():
        return "comfy"