"""
Project: LoLLMs
Author: ParisNeo
Description: Media classes:
    - WebcamImageSender: is a captures images from the webcam and sends them to a SocketIO client.
    - MusicPlayer: is a MusicPlayer class that allows you to play music using pygame library.
License: Apache 2.0
"""
from lollms.utilities import PackageManager
from lollms.com import LoLLMsCom
import platform
import subprocess

import os
import threading
if not PackageManager.check_package_installed("cv2"):
    os.system('sudo apt-get update')
    os.system('sudo apt-get install libgl1-mesa-glx python3-opencv -y')
import cv2



if not PackageManager.check_package_installed("scipy"):
    PackageManager.install_package("scipy")
    from scipy import signal
from scipy import signal

if not PackageManager.check_package_installed("matplotlib"):
    PackageManager.install_package("matplotlib")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

if not PackageManager.check_package_installed("whisper"):
    PackageManager.install_package("openai-whisper")
import whisper

from lollms.com import LoLLMsCom
import time
import json
import base64
import io
import numpy as np
if not PackageManager.check_package_installed("sounddevice"):
    PackageManager.install_package("sounddevice")
    PackageManager.install_package("wave")
import sounddevice as sd

class AudioRecorder:
    def __init__(self, socketio, filename, channels=1, sample_rate=16000, chunk_size=24678, silence_threshold=150.0, silence_duration=2, callback=None, lollmsCom=None):
        try:
            self.socketio = socketio
            self.filename = filename
            self.channels = channels
            self.sample_rate = sample_rate
            self.chunk_size = chunk_size
            self.audio_stream = None
            self.audio_frames = []
            self.is_recording = False
            self.silence_threshold = silence_threshold
            self.silence_duration = silence_duration
            self.last_sound_time = time.time()
            self.callback = callback
            self.lollmsCom = lollmsCom
            self.whisper_model = None
        except:
            self.socketio = socketio
            self.filename = filename
            self.channels = channels
            self.sample_rate = sample_rate
            self.chunk_size = chunk_size
            self.audio_stream = None
            self.audio_frames = []
            self.is_recording = False
            self.silence_threshold = silence_threshold
            self.silence_duration = silence_duration
            self.last_sound_time = time.time()
            self.callback = callback
            self.lollmsCom = lollmsCom
            self.whisper_model = None

    def start_recording(self):
        if self.whisper_model is None:
            self.lollmsCom.info("Loading whisper model")
            self.whisper_model=whisper.load_model("base.en")
        try:
            self.is_recording = True
            self.audio_stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self._record,
                blocksize=self.chunk_size
            )
            self.audio_stream.start()

            self.lollmsCom.info("Recording started...")
        except:
            self.lollmsCom.error("No audio input found!")

    def _record(self, indata, frames, time, status):
        first_recording = True  # Flag to track the first recording
        silence_duration = 5
        non_silent_start = None
        non_silent_end = None
        last_spectrogram_update = time.time()
        self.audio_frames = None
        buffered = np.array(indata)
        if self.audio_frames is not None:
            self.audio_frames = np.concatenate([self.audio_frames, buffered])
        else:
            self.audio_frames = buffered

        # Remove audio frames that are older than 30 seconds
        if len(self.audio_frames) > self.sample_rate * 30:
            self.audio_frames=self.audio_frames[-self.sample_rate * 30:]

        # Update spectrogram every 3 seconds
        if time.time() - last_spectrogram_update >= 1:
            self._update_spectrogram()
            last_spectrogram_update = time.time()

        # Check for silence
        rms = self._calculate_rms(buffered)
        if rms < self.silence_threshold:
            current_time = time.time()
            if current_time - self.last_sound_time >= silence_duration:
                if first_recording:
                    first_recording = False
                    silence_duration = self.silence_duration

                if self.callback and non_silent_start is not None and non_silent_end - non_silent_start >= 1:
                    self.lollmsCom.info("Analyzing")
                    # Convert to float
                    audio_data = self.audio_frames.astype(np.float32)
                    audio = wave.open(str(self.filename), 'wb')
                    audio.setnchannels(self.channels)
                    audio.setsampwidth(audio_stream.dtype.itemsize)
                    audio.setframerate(self.sample_rate)
                    audio.writeframes(b''.join(self.audio_frames[non_silent_start:non_silent_end]))
                    audio.close()

                    # Transcribe the audio using the whisper model
                    text = self.whisper_model.transcribe(audio_data[non_silent_start:non_silent_end])

                    self.callback(text)
                    print(text["text"])

                self.last_sound_time = time.time()
                non_silent_start = None

        else:
            self.last_sound_time = time.time()
            if non_silent_start is None:
                non_silent_start = len(self.audio_frames) - 1
            non_silent_end = len(self.audio_frames)

    def _update_spectrogram(self):
        audio_data = self.audio_frames[-self.sample_rate*30:]
        frequencies, _, spectrogram = signal.spectrogram(audio_data, self.sample_rate)

        # Generate a new times array that only spans the last 30 seconds
        times = np.linspace(0, 30, spectrogram.shape[1])

        # Plot spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(np.log(spectrogram), aspect='auto', origin='lower', cmap='inferno', extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')

        # Convert plot to base64 image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        # Send base64 image using socketio
        self.socketio.emit('update_spectrogram', img_base64)
        self.socketio.sleep(0.0)
        plt.close()

    def _calculate_rms(self, data):
        try:
            squared_sum = sum([sample ** 2 for sample in data])
            rms = np.sqrt(squared_sum / len(data))
        except:
            rms = 0
        return rms

    def stop_recording(self):
        self.is_recording = False
        if self.audio_stream:
            self.audio_stream.stop()
            import wave
            audio = wave.open(str(self.filename), 'wb')
            audio.setnchannels(self.channels)
            audio.setsampwidth(self.audio_stream.dtype.itemsize)
            audio.setframerate(self.sample_rate)
            audio.writeframes(b''.join(self.audio_frames))
            audio.close()

            self.lollmsCom.info(f"Recording saved to {self.filename}")
        else:
            self.warning("No recording available")

class WebcamImageSender:
    """
    Class for capturing images from the webcam and sending them to a SocketIO client.
    """

    def __init__(self, socketio, lollmsCom:LoLLMsCom=None):
        """
        Initializes the WebcamImageSender class.

        Args:
            socketio (socketio.Client): The SocketIO client object.
        """
        self.socketio = socketio
        self.last_image = None
        self.last_change_time = None
        self.capture_thread = None
        self.is_running = False
        self.lollmsCom = lollmsCom

    def start_capture(self):
        """
        Starts capturing images from the webcam in a separate thread.
        """
        self.is_running = True
        self.capture_thread = threading.Thread(target=self.capture_image)
        self.capture_thread.start()

    def stop_capture(self):
        """
        Stops capturing images from the webcam.
        """
        self.is_running = False
        self.capture_thread.join()

    def capture_image(self):
        """
        Captures images from the webcam, checks if the image content has changed, and sends the image to the client if it remains the same for 3 seconds.
        """
        try:
            cap = cv2.VideoCapture(0)

            while self.is_running:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.last_image is None or self.image_difference(gray) > 2:
                    self.last_image = gray
                    self.last_change_time = time.time()

                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer)
                self.socketio.emit("video_stream_image", image_base64.decode('utf-8'))

            cap.release()
        except:
            self.lollmsCom.error("Couldn't start webcam")

    def image_difference(self, image):
        """
        Calculates the difference between two images using the absolute difference method.

        Args:
            image (numpy.ndarray): The current image.

        Returns:
            int: The sum of pixel intensities representing the difference between the current image and the last image.
        """
        if self.last_image is None:
            return 0

        diff = cv2.absdiff(image, self.last_image)
        diff_sum = diff.sum()

        return diff_sum

class MusicPlayer(threading.Thread):
    """
    MusicPlayer class for playing music using pygame library.

    Attributes:
    - file_path (str): The path of the music file to be played.
    - paused (bool): Flag to indicate if the music is paused.
    - stopped (bool): Flag to indicate if the music is stopped.
    """

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.paused = False
        self.stopped = False

    def run(self):
        """
        The main function that runs in a separate thread to play the music.
        """
        if not PackageManager.check_package_installed("pygame"):
            PackageManager.install_package("pygame")
        import pygame

        pygame.mixer.init()
        pygame.mixer.music.load(self.file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy() and not self.stopped:
            if self.paused:
                pygame.mixer.music.pause()
            else:
                pygame.mixer.music.unpause()

    def pause(self):
        """
        Pauses the music.
        """
        self.paused = True

    def resume(self):
        """
        Resumes the paused music.
        """
        self.paused = False

    def stop(self):
        """
        Stops the music.
        """
        self.stopped = True
        pygame.mixer.music.stop()
