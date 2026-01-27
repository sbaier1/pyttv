import json
import logging
import urllib.parse
import urllib.request
import uuid

import requests
import websocket


class ComfyClient:
    def __init__(self, host):
        self.host = host
        self.client_id = str(uuid.uuid4())
        self.ws = websocket.WebSocket()
        self.ws_connected = False

    def connect(self):
        if not self.ws_connected:
            ws_url = f"ws://{self.host}/ws?clientId={self.client_id}"
            logging.info(f"Connecting to ComfyUI WebSocket: {ws_url}")
            self.ws.connect(ws_url)
            self.ws_connected = True

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.host}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def upload_image(self, image_data, filename, subfolder="", overwrite=True):
        """
        Uploads an image via POST /upload/image
        """
        files = {
            'image': (filename, image_data)
        }
        data = {
            'overwrite': str(overwrite).lower(),
            'subfolder': subfolder
        }
        response = requests.post(f"http://{self.host}/upload/image", files=files, data=data)
        response.raise_for_status()
        return response.json()

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.host}/view?{url_values}") as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen(f"http://{self.host}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def wait_for_completion(self, prompt_id):
        self.connect()
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        return True  # Execution done
            else:
                continue

    def get_images(self, prompt_id):
        output_images = []
        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                    output_images.append(image_data)
        return output_images
