"""
Client configuration and interaction with local LLama server.
"""

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin


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
