import pytest
from aiohttp import ClientError
from aioresponses import aioresponses

from kubevox.llama.llama_client import LlamaServerConfig, check_server_health


@pytest.fixture
def server_config():
    return LlamaServerConfig(host="localhost", port=8080)


@pytest.mark.asyncio
async def test_health_check_success(server_config):
    with aioresponses() as mocked:
        mocked.get("http://localhost:8080/health", status=200)
        is_healthy, message = await check_server_health(server_config)
        assert is_healthy is True
        assert message == "Server is healthy"


@pytest.mark.asyncio
async def test_health_check_failure(server_config):
    with aioresponses() as mocked:
        mocked.get("http://localhost:8080/health", status=500)
        is_healthy, message = await check_server_health(server_config)
        assert is_healthy is False
        assert message == "Server returned status code: 500"


@pytest.mark.asyncio
async def test_health_check_connection_error(server_config):
    with aioresponses() as mocked:
        mocked.get("http://localhost:8080/health", exception=ClientError())
        is_healthy, message = await check_server_health(server_config)
        assert is_healthy is False
        assert message.startswith("Failed to connect to server:")
