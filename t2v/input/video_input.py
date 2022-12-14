import logging
import subprocess as sp

import numpy as np

from t2v.config.root import RootConfig


class VideoInput:
    def __init__(self, cfg: RootConfig, video_path: str):
        """
        VideoInput reads frames from a given video file which can be used as init images for individual frame generation
        """
        command = ["ffmpeg",
                   '-i', video_path,
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-vcodec', 'rawvideo', '-']
        self.pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
        w, h = get_vid_dims(video_path)
        if w != cfg.width or h != cfg.height:
            raise RuntimeError(f"Actual width {w} and height {h} of the video file {video_path} are not matching "
                               f"the scenario width {cfg.width} and height {cfg.height}. Crop the video or make sure the settings are correct.")
        self.fps = get_vid_fps(video_path)
        self.w = w
        self.h = h
        self.video_path = video_path

    def next_frame(self):
        raw_image = self.pipe.stdout.read(self.w * self.h * 3)
        # transform the byte read into a numpy array
        image = np.frombuffer(raw_image, dtype='uint8')
        image = image.reshape((self.h, self.w, 3))
        # throw away the data in the pipe's buffer.
        self.pipe.stdout.flush()
        return image

    def skip_frame(self, n=1):
        logging.info(f"Skipping {n} frames of video file {self.video_path}")
        for i in range(0, n):
            # scroll the stdin buffer
            self.pipe.stdout.read(self.w * self.h * 3)
            # throw away the data in the pipe's buffer.
            self.pipe.stdout.flush()


def get_vid_dims(video_path):
    """
    Returns width,height as tuple for a given video file.
    Requires ffprobe in PATH

    :param video_path: path to video file
    :return: w,h tuple
    """
    line = ffprobe_params(video_path, "width,height")
    w, h = line.split("x")
    return int(w), int(h)


def ffprobe_params(video_path, param):
    command = ["ffprobe",
               '-v', 'error',
               '-select_streams', 'v',
               '-show_entries', f'stream={param}',
               '-of', 'csv=p=0:s=x',
               video_path]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
    line = pipe.stdout.readline(1024).decode("utf-8")
    return line


def get_vid_fps(video_path):
    """
    Returns frame rate of a given video file.
    Requires ffprobe in PATH

    :param video_path: path to video file
    :return: fps number
    """
    line = ffprobe_params(video_path, "r_frame_rate")
    num, den = line.split("/")
    return int(num)/int(den)
