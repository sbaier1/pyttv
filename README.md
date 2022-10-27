# text2video utility

a tool for leveraging text2image tools such as StableDiffusion for generating videos.

Mainly based on PyTTI and deforum, but adding and focusing around:

- designed around a CLI interface with well-structured and very versatile YAML configuration
- audio-reactivity and other multi-modality to come
- PyTTI style functions for most generative parameters
- extensible design for adding arbitrary mechanisms for generating animations

## usage

It's heavily advised to just use the API mechanism.

### installation

- prerequisite: Install python3.10+
- create venv: `python3 -m venv venv`
- source venv: `source venv/Scripts/activate` (or `source venv/bin/activate` on linux/macOS)
- install lean reqs for API mechanism: `pip install -r requirements.txt`
- get the remaining dependencies that don't come with wheels/proper build systems: `./init.sh` (or just copy the few `git clone` commands from the file on windows)

### general usage

- use https://github.com/AUTOMATIC1111/stable-diffusion-webui
- add the `--api` startup flag (in webui.sh or webui.bat or however you launch it) to expose the REST API (you can verify this by going to http://localhost:7860/docs and ensure there is a `/sdapi/...` endpoint there)
- Set your mechanism to `api` and configure the `host` parameter (if running locally it's always just http://localhost:7860)
- ensure web-ui is running
- check out the [examples](config) and build your config
- make sure the venv is sourced (see above)
- Run your scenario with `python3 -cp config -cn=yourconfig` where `-cp` specifies the path to your config directory and `-cn` specifies which config to run.

### Creating the final video

Once you have generated your frames, pyttv's job is done. Encoding to a video file is done by other tools.

A useful tool to use is [Flowframes](https://github.com/n00mkrad/flowframes) or similar tools that use RIFE interpolation or other decent interpolation mechanisms.

[rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) seems to work on macOS (M1). 

If you just want to encode your frames directly to a video file, you can of course use ffmpeg. `cd` into your output frame directory and run

```shell
cat *.png | ffmpeg -framerate 18 -f image2pipe -i - -c:v libx264 -pix_fmt yuv420p out.mp4
```
where `18` is to be replaced by your fps of course and `out.mp4` is the output filename.

### macOS notes

If you use an M1 mac, use `torch_device: cpu` in your configs. unfortunately the depth model currently does not work directly on the mps device.