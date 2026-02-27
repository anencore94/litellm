"""
Tests for auth_excluded_paths feature.

This feature allows users to configure paths that bypass authentication entirely
(including custom_auth) via the `auth_excluded_paths` setting in general_settings.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from starlette.datastructures import URL

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder


def _make_request(path: str) -> Request:
    """Create a mock Request with the given URL path."""
    request = Request(scope={"type": "http"})
    request._url = URL(url=path)
    return request


def _set_proxy_server_attrs(proxy_mod, overrides: dict) -> dict:
    """
    Set attributes on the proxy_server module and return the original values
    so they can be restored later.
    """
    defaults = {
        "prisma_client": None,
        "user_api_key_cache": MagicMock(),
        "proxy_logging_obj": MagicMock(),
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }
    defaults.update(overrides)

    original_values = {
        attr: getattr(proxy_mod, attr, None) for attr in defaults
    }
    for attr, val in defaults.items():
        setattr(proxy_mod, attr, val)

    return original_values


def _restore_proxy_server_attrs(proxy_mod, original_values: dict) -> None:
    """Restore original values on the proxy_server module."""
    for attr, val in original_values.items():
        setattr(proxy_mod, attr, val)


@pytest.mark.asyncio
async def test_auth_excluded_paths_bypasses_auth():
    """
    When a route is listed in auth_excluded_paths, the auth function should
    return a valid UserAPIKeyAuth without requiring an API key or calling
    custom_auth.
    """
    import litellm.proxy.proxy_server as _proxy_server_mod

    mock_custom_auth = AsyncMock()

    overrides = {
        "general_settings": {
            "auth_excluded_paths": ["/health", "/custom-no-auth"],
        },
        "user_custom_auth": mock_custom_auth,
    }

    original_values = _set_proxy_server_attrs(_proxy_server_mod, overrides)
    try:
        # Test /health path - should bypass auth
        request = _make_request("/health")
        result = await _user_api_key_auth_builder(
            request=request,
            api_key="",  # No API key provided
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )

        assert isinstance(result, UserAPIKeyAuth)
        assert result.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY

        # custom_auth should NOT have been called
        mock_custom_auth.assert_not_called()

        # Test /custom-no-auth path - should also bypass auth
        mock_custom_auth.reset_mock()
        request = _make_request("/custom-no-auth")
        result = await _user_api_key_auth_builder(
            request=request,
            api_key="",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )

        assert isinstance(result, UserAPIKeyAuth)
        assert result.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        mock_custom_auth.assert_not_called()
    finally:
        _restore_proxy_server_attrs(_proxy_server_mod, original_values)


@pytest.mark.asyncio
async def test_auth_excluded_paths_does_not_bypass_non_excluded_routes():
    """
    Routes NOT listed in auth_excluded_paths should still go through normal
    authentication. With a custom_auth set, the custom_auth function should
    be called for non-excluded routes.
    """
    import litellm.proxy.proxy_server as _proxy_server_mod

    mock_custom_auth = AsyncMock(
        return_value=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    )

    overrides = {
        "general_settings": {
            "auth_excluded_paths": ["/health"],
        },
        "user_custom_auth": mock_custom_auth,
    }

    original_values = _set_proxy_server_attrs(_proxy_server_mod, overrides)
    try:
        # /chat/completions is NOT excluded, so custom_auth should be called
        request = _make_request("/chat/completions")
        result = await _user_api_key_auth_builder(
            request=request,
            api_key="Bearer sk-some-key",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )

        assert isinstance(result, UserAPIKeyAuth)
        mock_custom_auth.assert_called_once()
    finally:
        _restore_proxy_server_attrs(_proxy_server_mod, original_values)


@pytest.mark.asyncio
async def test_auth_works_normally_without_auth_excluded_paths():
    """
    When auth_excluded_paths is not set (default), authentication should
    work exactly as before. The custom_auth function should be called.
    """
    import litellm.proxy.proxy_server as _proxy_server_mod

    mock_custom_auth = AsyncMock(
        return_value=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    )

    overrides = {
        "general_settings": {},  # No auth_excluded_paths
        "user_custom_auth": mock_custom_auth,
    }

    original_values = _set_proxy_server_attrs(_proxy_server_mod, overrides)
    try:
        request = _make_request("/health")
        result = await _user_api_key_auth_builder(
            request=request,
            api_key="Bearer sk-some-key",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )

        assert isinstance(result, UserAPIKeyAuth)
        # custom_auth SHOULD have been called since auth_excluded_paths is not set
        mock_custom_auth.assert_called_once()
    finally:
        _restore_proxy_server_attrs(_proxy_server_mod, original_values)


@pytest.mark.asyncio
async def test_auth_excluded_paths_empty_list():
    """
    When auth_excluded_paths is set to an empty list, no routes should be
    excluded from auth.
    """
    import litellm.proxy.proxy_server as _proxy_server_mod

    mock_custom_auth = AsyncMock(
        return_value=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    )

    overrides = {
        "general_settings": {
            "auth_excluded_paths": [],  # Empty list
        },
        "user_custom_auth": mock_custom_auth,
    }

    original_values = _set_proxy_server_attrs(_proxy_server_mod, overrides)
    try:
        request = _make_request("/health")
        result = await _user_api_key_auth_builder(
            request=request,
            api_key="Bearer sk-some-key",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )

        assert isinstance(result, UserAPIKeyAuth)
        # custom_auth SHOULD have been called since no paths are excluded
        mock_custom_auth.assert_called_once()
    finally:
        _restore_proxy_server_attrs(_proxy_server_mod, original_values)


@pytest.mark.asyncio
async def test_auth_excluded_paths_exact_match_only():
    """
    auth_excluded_paths should only match exact paths. A path like /health
    should not match /health/detailed.

    We test this by verifying custom_auth IS called for /chat/completions
    (a non-admin route) when only /health is excluded.
    """
    import litellm.proxy.proxy_server as _proxy_server_mod

    mock_custom_auth = AsyncMock(
        return_value=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)
    )

    overrides = {
        "general_settings": {
            "auth_excluded_paths": ["/health"],
        },
        "user_custom_auth": mock_custom_auth,
    }

    original_values = _set_proxy_server_attrs(_proxy_server_mod, overrides)
    try:
        # /chat/completions is NOT an exact match for /health, so auth should proceed
        request = _make_request("/chat/completions")
        result = await _user_api_key_auth_builder(
            request=request,
            api_key="Bearer sk-some-key",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )

        assert isinstance(result, UserAPIKeyAuth)
        # custom_auth should be called since /chat/completions is not excluded
        mock_custom_auth.assert_called_once()

        # But /health should be excluded (verified: no custom_auth call)
        mock_custom_auth.reset_mock()
        request = _make_request("/health")
        result = await _user_api_key_auth_builder(
            request=request,
            api_key="",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )
        assert isinstance(result, UserAPIKeyAuth)
        assert result.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        mock_custom_auth.assert_not_called()
    finally:
        _restore_proxy_server_attrs(_proxy_server_mod, original_values)
