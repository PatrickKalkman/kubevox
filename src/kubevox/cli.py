"""
Command-line interface for KubeVox.
"""

import asyncio
import sys

from loguru import logger

# Configure logger to only show info and higher
logger.remove()
logger.add(sys.stderr, level="INFO")

from kubevox.llama.llama_client import LlamaClient, LlamaServerConfig
from kubevox.registry.function_registry import FunctionRegistry


async def main():
    """Main entry point for the CLI."""
    logger.info("Initializing LlamaClient...")

    # Initialize the client with default configuration
    config = LlamaServerConfig()
    client = LlamaClient(config)

    # Check server health first
    healthy, message = await client.check_server_health()
    if not healthy:
        logger.error(f"Server health check failed: {message}")
        return

    logger.info("Server is healthy, sending query...")

    # Log available functions
    logger.debug(f"Available functions: {[f.__name__ for f in FunctionRegistry.functions]}")

    try:
        response = await client.generate_llm_response("Get the number of namespaces in the Kubernetes cluster")
        logger.info("Response received:")
        logger.info(response)

        function_calls = client.extract_function_calls(response)
        logger.info("Extracted function calls:")
        logger.info(function_calls)
    except Exception as e:
        logger.error(f"Error generating response: {e}")


if __name__ == "__main__":
    asyncio.run(main())
