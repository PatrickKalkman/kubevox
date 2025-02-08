"""
Client configuration and interaction with local LLama server.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientError

from kubevox.llama.llama_tools import generate_assistant_header, generate_system_prompt, generate_user_message


@dataclass
class LlamaServerConfig:
    """Configuration for local LLama server connection."""

    host: str = "localhost"
    port: int = 8080
    model_path: Optional[str] = None
    n_ctx: int = 2048  # Context window size
    n_gpu_layers: int = 0  # Number of layers to offload to GPU
    seed: int = -1  # RNG seed, -1 for random

    @property
    def base_url(self) -> str:
        """Get the base URL for the LLama server."""
        return f"http://{self.host}:{self.port}"


async def check_server_health(config: LlamaServerConfig) -> Tuple[bool, str]:
    """
    Check if the LLama server is running and healthy.

    Returns:
        Tuple of (is_healthy: bool, message: str)
    """
    try:
        health_url = urljoin(config.base_url, "/health")
        async with aiohttp.ClientSession() as session:
            async with session.get(health_url, timeout=5.0) as response:
                if response.status == 200:
                    return True, "Server is healthy"
                else:
                    return False, f"Server returned status code: {response.status}"

    except ClientError as e:
        return False, f"Failed to connect to server: {str(e)}"
    except asyncio.TimeoutError:
        return False, "Connection timed out"


async def generate_llm_response(
    config: LlamaServerConfig,
    user_message: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    stop: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a response from the Llama server using the provided user message.

    Args:
        config: Server configuration
        user_message: The user's input message
        temperature: Sampling temperature (default: 0.7)
        top_p: Nucleus sampling threshold (default: 0.9)
        max_tokens: Maximum number of tokens to generate (default: 2048)
        stop: Optional list of strings to stop generation at

    Returns:
        Dictionary containing the model's response
    """
    try:
        completion_url = urljoin(config.base_url, "/completion")

        # Construct the full prompt with system context and formatting
        full_prompt = (
            f"{generate_system_prompt()}\n{generate_user_message(user_message)}\n{generate_assistant_header()}"
        )

        payload = {
            "prompt": full_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "n_predict": max_tokens,
            "stop": stop or [],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(completion_url, json=payload, timeout=30.0) as response:
                if response.status != 200:
                    raise ClientError(f"Server returned status code: {response.status}")
                return await response.json()

    except (ClientError, asyncio.TimeoutError) as e:
        raise ClientError(f"Failed to get completion: {str(e)}")
