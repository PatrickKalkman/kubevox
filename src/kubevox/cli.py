"""
Command-line interface for KubeVox.
"""

import asyncio
import sys
from typing import Optional

from kubevox.utils.timing import timing

import typer
from dotenv import load_dotenv
from loguru import logger

from kubevox.assistant import Assistant
from kubevox.llama.llama_client import LlamaClient, LlamaServerConfig

# Configure logger to only show info and higher
logger.remove()
logger.add(sys.stderr, level="INFO")

app = typer.Typer(help="Kubernetes Voice Assistant CLI")

load_dotenv()


async def run_text_mode(assistant: Assistant, query: str) -> None:
    """Run the assistant in text mode with a single query.

    Args:
        assistant: Initialized Assistant instance
        query: Text query to process
    """
    with timing("Total Query Processing"):
        with timing("Query Processing"):
            response = await assistant.process_query(query)
        
        with timing("Response Formatting"):
            formatted_responses = [
                result.get("formatted_response", "")
                for result in response["results"]
                if result.get("formatted_response")
            ]
        
        if formatted_responses:
            combined_response = " and ".join(formatted_responses)
            with timing("Response Output"):
                if assistant.output_mode == "voice" and assistant.speaker:
                    assistant.speaker.speak(combined_response)
                else:
                    logger.info(f"Assistant: {combined_response}")
        else:
            logger.warning("No formatted responses available")


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


@app.command()
def text(
    query: str = typer.Argument(..., help="Text query to process"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    output: str = typer.Option("text", "--output", help="Output mode (text or voice via ElevenLabs)"),
    elevenlabs_key: Optional[str] = typer.Option(None, "--elevenlabs-key", help="ElevenLabs API key"),
    model: str = typer.Option(
        "mlx-community/whisper-large-v3-turbo", "--model", help="Path or name of the Whisper model to use"
    ),
):
    """Run in text mode with a single query."""

    async def run():
        logger.info("Initializing LlamaClient...")
        config = LlamaServerConfig()
        client = LlamaClient(config)

        healthy, message = await client.check_server_health()
        if not healthy:
            logger.error(f"Server health check failed: {message}")
            raise typer.Exit(1)

        logger.info("Server is healthy, starting assistant...")

        assistant = Assistant(
            llamaClient=client,
            model_path=model,
            output_mode=output,
            elevenlabs_api_key=elevenlabs_key,
        )

        try:
            await run_text_mode(assistant, query)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise typer.Exit(1)

    asyncio.run(run())


@app.command()
def voice(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    output: str = typer.Option("text", "--output", help="Output mode (text or voice via ElevenLabs)"),
    elevenlabs_key: Optional[str] = typer.Option(None, "--elevenlabs-key", help="ElevenLabs API key"),
    model: str = typer.Option(
        "mlx-community/whisper-large-v3-turbo", "--model", help="Path or name of the Whisper model to use"
    ),
    duration: float = typer.Option(4.0, "--duration", help="Recording duration in seconds"),
    device: Optional[int] = typer.Option(None, "--device", help="Audio input device index"),
):
    """Run in voice interaction mode."""
    logger.info("Initializing LlamaClient...")
    config = LlamaServerConfig()
    client = LlamaClient(config)

    async def check_health():
        healthy, message = await client.check_server_health()
        if not healthy:
            logger.error(f"Server health check failed: {message}")
            raise typer.Exit(1)

    asyncio.run(check_health())

    logger.info("Server is healthy, starting assistant...")

    assistant = Assistant(
        llamaClient=client,
        model_path=model,
        input_device=device,
        recording_duration=duration,
        output_mode=output,
        elevenlabs_api_key=elevenlabs_key,
    )

    try:
        run_voice_mode(assistant, duration, device)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
