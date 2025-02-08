"""
Client configuration and interaction with local LLama server.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientError


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
