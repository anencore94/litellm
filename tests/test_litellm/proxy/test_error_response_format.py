"""
Tests for the error_response_format general_settings option.

Verifies that ProxyException responses can be returned in either
OpenAI format (default) or FastAPI format based on configuration.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import ProxyException


class TestOpenAIExceptionHandlerFormat:
    """Tests for the openai_exception_handler with error_response_format."""

    @pytest.mark.asyncio
    async def test_default_format_is_openai(self):
        """When error_response_format is not set, responses use OpenAI format."""
        from litellm.proxy.proxy_server import openai_exception_handler

        exc = ProxyException(
            message="Test error message",
            type="invalid_request_error",
            param="test_param",
            code=400,
        )
        mock_request = MagicMock()

        with patch("litellm.proxy.proxy_server.general_settings", {}):
            response = await openai_exception_handler(mock_request, exc)

        assert response.status_code == 400
        import json

        body = json.loads(response.body)
        assert "error" in body
        assert body["error"]["message"] == "Test error message"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["param"] == "test_param"
        assert body["error"]["code"] == "400"

    @pytest.mark.asyncio
    async def test_explicit_openai_format(self):
        """When error_response_format is 'openai', responses use OpenAI format."""
        from litellm.proxy.proxy_server import openai_exception_handler

        exc = ProxyException(
            message="Test error message",
            type="invalid_request_error",
            param="test_param",
            code=400,
        )
        mock_request = MagicMock()

        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"error_response_format": "openai"},
        ):
            response = await openai_exception_handler(mock_request, exc)

        assert response.status_code == 400
        import json

        body = json.loads(response.body)
        assert "error" in body
        assert body["error"]["message"] == "Test error message"
        assert "detail" not in body

    @pytest.mark.asyncio
    async def test_fastapi_format(self):
        """When error_response_format is 'fastapi', responses use FastAPI format."""
        from litellm.proxy.proxy_server import openai_exception_handler

        exc = ProxyException(
            message="Test error message",
            type="invalid_request_error",
            param="test_param",
            code=400,
        )
        mock_request = MagicMock()

        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"error_response_format": "fastapi"},
        ):
            response = await openai_exception_handler(mock_request, exc)

        assert response.status_code == 400
        import json

        body = json.loads(response.body)
        assert "detail" in body
        assert body["detail"] == "Test error message"
        assert "error" not in body

    @pytest.mark.asyncio
    async def test_fastapi_format_preserves_status_code(self):
        """FastAPI format preserves the original HTTP status code."""
        from litellm.proxy.proxy_server import openai_exception_handler

        exc = ProxyException(
            message="Rate limit exceeded",
            type="rate_limit_error",
            param=None,
            code=429,
        )
        mock_request = MagicMock()

        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"error_response_format": "fastapi"},
        ):
            response = await openai_exception_handler(mock_request, exc)

        assert response.status_code == 429
        import json

        body = json.loads(response.body)
        assert body["detail"] == "Rate limit exceeded"

    @pytest.mark.asyncio
    async def test_fastapi_format_preserves_custom_headers(self):
        """FastAPI format preserves custom headers from the exception."""
        from litellm.proxy.proxy_server import openai_exception_handler

        exc = ProxyException(
            message="Custom header test",
            type="invalid_request_error",
            param=None,
            code=400,
            headers={"x-custom-header": "test-value"},
        )
        mock_request = MagicMock()

        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"error_response_format": "fastapi"},
        ):
            response = await openai_exception_handler(mock_request, exc)

        assert response.status_code == 400
        assert response.headers.get("x-custom-header") == "test-value"
        import json

        body = json.loads(response.body)
        assert body["detail"] == "Custom header test"


class TestErrorResponseFormatConfigSchema:
    """Tests for the error_response_format field in ConfigGeneralSettings."""

    def test_config_general_settings_accepts_openai(self):
        from litellm.proxy._types import ConfigGeneralSettings

        config = ConfigGeneralSettings(error_response_format="openai")
        assert config.error_response_format == "openai"

    def test_config_general_settings_accepts_fastapi(self):
        from litellm.proxy._types import ConfigGeneralSettings

        config = ConfigGeneralSettings(error_response_format="fastapi")
        assert config.error_response_format == "fastapi"

    def test_config_general_settings_default_is_none(self):
        from litellm.proxy._types import ConfigGeneralSettings

        config = ConfigGeneralSettings()
        assert config.error_response_format is None

    def test_config_general_settings_rejects_invalid_value(self):
        from pydantic import ValidationError

        from litellm.proxy._types import ConfigGeneralSettings

        with pytest.raises(ValidationError):
            ConfigGeneralSettings(error_response_format="invalid")
