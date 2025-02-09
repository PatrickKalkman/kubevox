import queue
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import mlx_whisper
import numpy as np
import sounddevice as sd
from loguru import logger
from pynput import keyboard
from scipy import signal


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    channels: int = 1
    recording_duration: float = 5.0
    min_amplitude: float = 0.01
    noise_reduction: bool = True
    block_duration_ms: int = 100  # Duration of audio blocks in milliseconds


class AudioProcessor:
    """Handles audio signal processing operations."""

    def __init__(self, config: AudioConfig):
        self.config = config

    def resample(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio data to the target sample rate."""
        duration = len(audio_data) / orig_sr
        target_length = int(duration * target_sr)
        logger.debug(f"Resampling {len(audio_data)} samples to {target_length}")
        return signal.resample(audio_data, target_length)

    def normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to the range [-1, 1]."""
        if len(audio_data) == 0:
            logger.warning("Received empty audio data for normalization")
            return np.array([])

        audio_data = audio_data.flatten()
        max_amplitude = np.max(np.abs(audio_data))

        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude
            logger.debug(f"Normalized audio: max amplitude = {np.max(np.abs(audio_data))}")
        else:
            logger.warning("Audio data is silent (max amplitude = 0)")

        return audio_data

    def reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction to the audio signal."""
        if len(audio_data) == 0:
            return audio_data

        # Estimate noise from the first 0.1 seconds
        noise_sample = audio_data[: int(self.config.sample_rate * 0.1)]
        noise_profile = np.mean(np.abs(noise_sample))
        logger.debug(f"Estimated noise profile: {noise_profile}")

        # Apply noise gate
        threshold = noise_profile * 2
        audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)

        # Apply low-pass filter
        b, a = signal.butter(4, 2000 / (self.config.sample_rate / 2), btype="low")
        return signal.filtfilt(b, a, audio_data)


class AudioDeviceManager:
    """Manages audio device configuration and streaming."""

    def __init__(self, config: AudioConfig, device_index: Optional[int] = None):
        self.config = config
        self.device_index = device_index
        self.device_sample_rate = self._get_device_sample_rate()
        self.stream: Optional[sd.InputStream] = None

    def _get_device_sample_rate(self) -> int:
        """Get the sample rate of the selected audio device."""
        try:
            device_info = sd.query_devices(self.device_index, "input")
            logger.info(f"Using audio input device: {device_info['name']}")
            return int(device_info["default_samplerate"])
        except sd.PortAudioError as e:
            logger.error(f"Error accessing audio device: {e}")
            raise

    def create_stream(self, callback: Callable) -> sd.InputStream:
        """Create and configure the audio input stream."""
        try:
            blocksize = int(self.device_sample_rate * self.config.block_duration_ms / 1000)
            return sd.InputStream(
                device=self.device_index,
                channels=self.config.channels,
                samplerate=self.device_sample_rate,
                blocksize=blocksize,
                callback=callback,
            )
        except sd.PortAudioError as e:
            logger.error(f"Error initializing audio stream: {e}")
            raise


class TranscriptionResult(Protocol):
    """Protocol for transcription results."""
    text: str


class WhisperTranscriber:
    def __init__(
        self,
        model_path: str = "mlx-community/whisper-large-v3-turbo",
        sample_rate: int = 16000,
        channels: int = 1,
        recording_duration: float = 5.0,
        input_device: Optional[int] = None,
        min_amplitude: float = 0.01,
        noise_reduction: bool = True,
    ):
        """Initialize the WhisperTranscriber."""
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording_duration = recording_duration
        self.input_device = input_device
        self.min_amplitude = min_amplitude
        self.noise_reduction = noise_reduction

        # State management
        self._is_recording = False
        self._is_listening = False
        self._callback = None
        self._audio_queue = queue.Queue()
        self._recording_thread = None

        # Initialize audio stream
        self._verify_audio_device()
        self._init_audio_stream()

    def _verify_audio_device(self) -> None:
        """Verify that the selected audio input device is working."""
        try:
            device_info = sd.query_devices(self.input_device, "input")
            logger.info(f"Using audio input device: {device_info['name']}")
            self.device_sample_rate = int(device_info["default_samplerate"])
            logger.info(f"Device sample rate: {self.device_sample_rate}Hz")

            if self.device_sample_rate != self.sample_rate:
                logger.info(f"Will resample from {self.device_sample_rate}Hz to {self.sample_rate}Hz")
        except sd.PortAudioError as e:
            logger.error(f"Error accessing audio device: {e}")
            raise

    def _init_audio_stream(self):
        """Initialize the audio input stream."""
        try:
            self.stream = sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.device_sample_rate,
                blocksize=int(self.device_sample_rate * 0.1),  # 100ms blocks
                callback=self._audio_callback,
            )
        except sd.PortAudioError as e:
            logger.error(f"Error initializing audio stream: {e}")
            raise

    def _audio_callback(self, indata, frames, time, status):
        """Callback for the audio stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self._is_recording:
            self._audio_queue.put(indata.copy())

    def _resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio data to the target sample rate."""
        duration = len(audio_data) / orig_sr
        target_length = int(duration * target_sr)
        logger.debug(f"Resampling {len(audio_data)} samples to {target_length}")
        resampled = signal.resample(audio_data, target_length)
        return resampled

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to the range [-1, 1]."""
        if len(audio_data) == 0:
            logger.warning("Received empty audio data for normalization")
            return np.array([])

        audio_data = audio_data.flatten()
        max_amplitude = np.max(np.abs(audio_data))

        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude
            logger.debug(f"Normalized audio: max amplitude = {np.max(np.abs(audio_data))}")
        else:
            logger.warning("Audio data is silent (max amplitude = 0)")

        return audio_data

    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction to the audio signal."""
        if len(audio_data) == 0:
            return audio_data

        # Estimate noise from the first 0.1 seconds
        noise_sample = audio_data[: int(self.sample_rate * 0.1)]
        noise_profile = np.mean(np.abs(noise_sample))
        logger.debug(f"Estimated noise profile: {noise_profile}")

        # Apply noise gate
        threshold = noise_profile * 2
        audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)

        # Apply low-pass filter
        b, a = signal.butter(4, 2000 / (self.sample_rate / 2), btype="low")
        filtered_audio = signal.filtfilt(b, a, audio_data)

        return filtered_audio

    def start_recording(self):
        """Start recording audio."""
        logger.info("Starting recording...")
        # Clear any old audio data
        while not self._audio_queue.empty():
            self._audio_queue.get()

        self._is_recording = True
        self.stream.start()

    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and process the audio."""
        logger.info("Stopping recording...")
        self._is_recording = False
        self.stream.stop()

        # Collect all audio data from the queue
        audio_chunks = []
        while not self._audio_queue.empty():
            audio_chunks.append(self._audio_queue.get())

        if not audio_chunks:
            logger.warning("No audio data collected")
            return None

        # Combine audio chunks
        audio_data = np.concatenate(audio_chunks)

        # Process audio
        logger.info(f"Processing audio data: shape={audio_data.shape}, dtype={audio_data.dtype}")

        # Resample if necessary
        if self.device_sample_rate != self.sample_rate:
            logger.info(f"Resampling from {self.device_sample_rate}Hz to {self.sample_rate}Hz")
            audio_data = self._resample_audio(audio_data, self.device_sample_rate, self.sample_rate)

        # Normalize
        audio_data = self._normalize_audio(audio_data)

        # Check amplitude
        max_amplitude = np.max(np.abs(audio_data))
        logger.info(f"Max amplitude: {max_amplitude}")
        if max_amplitude < self.min_amplitude:
            logger.warning("Audio input level too low")
            return None

        # Apply noise reduction if enabled
        if self.noise_reduction:
            audio_data = self._apply_noise_reduction(audio_data)
            audio_data = self._normalize_audio(audio_data)

        return audio_data.astype(np.float32)

    def transcribe_audio(self, audio_data: np.ndarray) -> dict:
        """Transcribe audio data using mlx-whisper."""
        try:
            logger.info(f"Starting transcription... Audio shape: {audio_data.shape}")
            logger.debug(
                f"Audio stats - min: {np.min(audio_data)}, max: {np.max(audio_data)}, mean: {np.mean(audio_data)}"
            )

            result = mlx_whisper.transcribe(audio_data, path_or_hf_repo=self.model_path)
            if result is None:
                return {"text": "Error during transcription"}
            if not result.get("text"):
                return {"text": ""}
            return result
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return {"text": f"Error during transcription: {str(e)}"}

    def on_press(self, key):
        """Handle key press events."""
        try:
            if key == keyboard.Key.space and not self._is_recording:
                logger.info("Space pressed - starting recording")
                self.start_recording()
            return True
        except Exception as e:
            logger.error(f"Error during key press: {e}")
            return True

    def on_release(self, key):
        """Handle key release events."""
        logger.debug(f"Key released: {key}")
        try:
            if key == keyboard.Key.space and self._is_recording:
                logger.info("Space released - stopping recording")
                audio_data = self.stop_recording()

                if audio_data is not None:
                    logger.info("Processing recorded audio...")
                    transcription_result = self.transcribe_audio(audio_data)
                    transcribed_text = transcription_result.get("text", "")
                    if self._callback:
                        self._callback(transcribed_text)
                    else:
                        print(f"Transcription: {transcribed_text}")

            elif key == keyboard.Key.esc:
                logger.info("Escape pressed - stopping listener")
                self._is_listening = False
                return False
            return True
        except Exception as e:
            logger.error(f"Error during key release: {e}")
            return True

    def start_listening(self, callback: Optional[Callable[[str], None]] = None):
        """Start listening for keyboard events to trigger recording."""
        self._is_listening = True
        self._callback = callback
        logger.info("Started listening for input (Press and hold spacebar to record, ESC to quit)")

        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def stop_listening(self):
        """Stop the continuous listening loop."""
        self._is_listening = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        logger.info("Stopped listening")

    def set_input_device(self, device_index: int):
        """Set the audio input device by its index."""
        logger.info(f"Changing input device to index: {device_index}")
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

        self.input_device = device_index
        self._verify_audio_device()
        self._init_audio_stream()
        logger.info(f"Input device changed to index: {device_index}")
