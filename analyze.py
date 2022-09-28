from librosa.beat import beat_track
import numpy as np
import subprocess

SAMPLERATE=44100
input_audio = 'E:\\Downloads\\01-02-Jon_Hopkins-Open_Eye_Signal-SMR.flac'


pipe = subprocess.Popen(['ffmpeg', '-i', input_audio,
                         '-f', 's16le',
                         '-acodec', 'pcm_s16le',
                         '-ar', str(SAMPLERATE),
                         '-ac', '1',
                         '-'], stdout=subprocess.PIPE, bufsize=10 ** 8)

audio_samples = np.array([], dtype=np.float32)

# read the audio file from the pipe in 0.5s blocks (2 bytes per sample)
while True:
    buf = pipe.stdout.read(SAMPLERATE)
    audio_samples = np.append(audio_samples, np.frombuffer(buf, dtype=np.int16)/32767)
    if len(buf) < SAMPLERATE:
        break
if len(audio_samples) < 0:
    raise RuntimeError("Audio samples are empty, assuming load failed")

beats = beat_track(y=audio_samples, sr=44100, units='time', bpm=122.0)

print(f"beats: {beats}")
