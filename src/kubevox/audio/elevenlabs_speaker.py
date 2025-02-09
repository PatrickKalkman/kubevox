import os
from typing import Any, Optional

from elevenlabs import ElevenLabs, stream


class ElevenLabsSpeaker:
    """
    A class for converting text to speech using the ElevenLabs API.

    Attributes:
        client (ElevenLabs): The ElevenLabs client instance.
        default_voice_id (str): Default voice ID to use for synthesis.
        default_model_id (str): Default model ID to use for synthesis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        default_model_id: str = "eleven_multilingual_v2",
    ):
        """Initialize the ElevenLabsSpeaker.

        Args:
            api_key: Optional API key for ElevenLabs. If not provided, reads from ELEVENLABS_API_KEY env var.
            default_voice_id: Default voice ID to use for speech synthesis.
            default_model_id: Default model ID to use for speech synthesis.
        """
        self.client = ElevenLabs(api_key=api_key or os.environ.get("ELEVENLABS_API_KEY"))
        self.default_voice_id = default_voice_id
        self.default_model_id = default_model_id

    def speak(
        self, text: str, voice_id: Optional[str] = None, model_id: Optional[str] = None, stream_audio: bool = True
    ) -> Optional[Any]:
        """Convert text to speech and optionally stream it immediately.

        Args:
            text: The text to convert to speech.
            voice_id: Optional voice ID to use. Falls back to default if not provided.
            model_id: Optional model ID to use. Falls back to default if not provided.
            stream_audio: Whether to stream the audio immediately (True) or return the stream (False).

        Returns:
            None if stream_audio is True, otherwise returns the audio stream.
        """
        audio_stream = self.client.text_to_speech.convert_as_stream(
            text=text, voice_id=voice_id or self.default_voice_id, model_id=model_id or self.default_model_id
        )

        if stream_audio:
            stream(audio_stream)
            return None

        return audio_stream
