"""
Tests for the LlamaClient functionality.
"""

import pytest
import pytest_asyncio
from aiohttp import ClientError

from kubevox.llama.llama_client import LlamaServerConfig


class MockResponse:
    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockClientSession:
    def __init__(self, mock_response):
        self.mock_response = mock_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get(self, url, **kwargs):
        return self.mock_response


@pytest.mark.asyncio
async def test_check_health_success(monkeypatch):
    """Test successful health check."""
    config = LlamaServerConfig()
    mock_response = MockResponse(status=200)

    async def mock_client_session(*args, **kwargs):
        return MockClientSession(mock_response)

    monkeypatch.setattr("aiohttp.ClientSession", mock_client_session)

    is_healthy, message = await config.check_health()
    assert is_healthy is True
    assert message == "Server is healthy"


@pytest.mark.asyncio
async def test_check_health_error_status(monkeypatch):
    """Test health check with error status code."""
    config = LlamaServerConfig()
    mock_response = MockResponse(status=500)

    async def mock_client_session(*args, **kwargs):
        return MockClientSession(mock_response)

    monkeypatch.setattr("aiohttp.ClientSession", mock_client_session)

    is_healthy, message = await config.check_health()
    assert is_healthy is False
    assert "500" in message


@pytest.mark.asyncio
async def test_check_health_connection_error(monkeypatch):
    """Test health check with connection error."""
    config = LlamaServerConfig()

    async def mock_client_session(*args, **kwargs):
        raise ClientError()

    monkeypatch.setattr("aiohttp.ClientSession", mock_client_session)

    is_healthy, message = await config.check_health()
    assert is_healthy is False
    assert "Failed to connect" in message
