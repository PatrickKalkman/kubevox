"""
Command-line interface for KubeVox.
"""

import argparse
import asyncio
import sys
from typing import Optional

from loguru import logger

from kubevox.assistant import Assistant
from kubevox.llama.llama_client import LlamaClient, LlamaServerConfig

# Configure logger to only show info and higher
logger.remove()
logger.add(sys.stderr, level="INFO")


async def run_text_mode(assistant: Assistant, query: str) -> None:
    """Run the assistant in text mode with a single query.

    Args:
        assistant: Initialized Assistant instance
        query: Text query to process
    """
    response = await assistant.process_query(query)
    response_text = response.get("response", response)
    if assistant.output_mode == "voice" and assistant.speaker:
        assistant.speaker.speak(response_text)
    else:
        print(f"Assistant: {response_text}")


def run_voice_mode(assistant: Assistant, duration: float, device_index: Optional[int]) -> None:
    """Run the assistant in voice interaction mode.

    Args:
        assistant: Initialized Assistant instance
        duration: Recording duration in seconds
        device_index: Audio input device index
    """
    if device_index is not None:
        assistant.set_input_device(device_index)

    try:
        assistant.start_voice_interaction()
    except KeyboardInterrupt:
        print("\nStopping voice interaction...")
    finally:
        assistant.stop_voice_interaction()


async def main():
    parser = argparse.ArgumentParser(
        description="Kubernetes Voice Assistant CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General options
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    # Input mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-t", "--text", help="Run in text mode with the provided query")
    mode_group.add_argument("--voice", action="store_true", help="Run in voice interaction mode")

    # Output mode selection
    parser.add_argument(
        "--output", choices=["text", "voice"], default="text", help="Choose output mode (text or voice via ElevenLabs)"
    )
    parser.add_argument("--elevenlabs-key", help="ElevenLabs API key (can also be set via ELEVENLABS_API_KEY env var)")

    # Voice mode options
    parser.add_argument(
        "--model", default="mlx-community/whisper-large-v3-turbo", help="Path or name of the Whisper model to use"
    )
    parser.add_argument("--duration", type=float, default=4.0, help="Recording duration in seconds for voice mode")
    parser.add_argument("--device", type=int, help="Audio input device index")

    args = parser.parse_args()

    logger.info("Initializing LlamaClient...")

    # Initialize the client with default configuration
    config = LlamaServerConfig()
    client = LlamaClient(config)

    # Check server health first
    healthy, message = await client.check_server_health()
    if not healthy:
        logger.error(f"Server health check failed: {message}")
        return

    logger.info("Server is healthy, starting assistant...")

    assistant = Assistant(
        llamaClient=client,
        model_path=args.model,
        input_device=args.device,
        recording_duration=args.duration,
        output_mode=args.output,
        elevenlabs_api_key=args.elevenlabs_key,
    )

    # Run in selected mode
    try:
        if args.text:
            await run_text_mode(assistant, args.text)
        else:  # voice mode
            run_voice_mode(assistant, args.duration, args.device)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    asyncio.run(main())
