"""
Tests for flatten_provider_specific_fields setting.

Verifies that when litellm.flatten_provider_specific_fields is True,
provider_specific_fields dicts are flattened into their parent objects
in the serialized response.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy.utils import model_dump_with_preserved_fields
from litellm.types.utils import Choices, Message, ModelResponse, Usage


def _make_response_with_provider_fields() -> ModelResponse:
    """Create a ModelResponse with provider_specific_fields on the message."""
    msg = Message(
        content="Hello",
        role="assistant",
        provider_specific_fields={
            "content_filter_results": {"hate": {"filtered": False}},
            "refusal": None,
        },
    )
    choice = Choices(
        finish_reason="stop",
        index=0,
        message=msg,
    )
    response = ModelResponse(
        id="chatcmpl-test",
        choices=[choice],
        model="gpt-4",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    return response


class TestFlattenProviderSpecificFields:
    """Tests for the flatten_provider_specific_fields setting."""

    def test_default_keeps_provider_specific_fields_nested(self):
        """By default, provider_specific_fields stays as a nested dict."""
        original = litellm.flatten_provider_specific_fields
        litellm.flatten_provider_specific_fields = False
        try:
            response = _make_response_with_provider_fields()
            result = model_dump_with_preserved_fields(response)

            msg = result["choices"][0]["message"]
            assert "provider_specific_fields" in msg
            assert isinstance(msg["provider_specific_fields"], dict)
            assert "content_filter_results" in msg["provider_specific_fields"]
            # Fields should NOT be at the top level of message
            assert "content_filter_results" not in msg
        finally:
            litellm.flatten_provider_specific_fields = original

    def test_flatten_moves_fields_to_parent(self):
        """When enabled, provider_specific_fields are flattened into parent."""
        original = litellm.flatten_provider_specific_fields
        litellm.flatten_provider_specific_fields = True
        try:
            response = _make_response_with_provider_fields()
            result = model_dump_with_preserved_fields(response)

            msg = result["choices"][0]["message"]
            # provider_specific_fields key should be removed
            assert "provider_specific_fields" not in msg
            # Fields should be at the message level
            assert "content_filter_results" in msg
            assert msg["content_filter_results"] == {"hate": {"filtered": False}}
        finally:
            litellm.flatten_provider_specific_fields = original

    def test_flatten_does_not_overwrite_existing_fields(self):
        """Flattened fields should not overwrite existing message fields."""
        original = litellm.flatten_provider_specific_fields
        litellm.flatten_provider_specific_fields = True
        try:
            msg = Message(
                content="Hello",
                role="assistant",
                provider_specific_fields={
                    "custom_field": "custom_value",
                },
            )
            choice = Choices(finish_reason="stop", index=0, message=msg)
            response = ModelResponse(
                id="chatcmpl-test",
                choices=[choice],
                model="gpt-4",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

            result = model_dump_with_preserved_fields(response)
            msg_dict = result["choices"][0]["message"]

            # content and role should be preserved
            assert msg_dict["content"] == "Hello"
            assert msg_dict["role"] == "assistant"
            # custom field should be flattened
            assert msg_dict["custom_field"] == "custom_value"
            assert "provider_specific_fields" not in msg_dict
        finally:
            litellm.flatten_provider_specific_fields = original

    def test_flatten_handles_no_provider_specific_fields(self):
        """When there are no provider_specific_fields, flatten has no effect."""
        original = litellm.flatten_provider_specific_fields
        litellm.flatten_provider_specific_fields = True
        try:
            msg = Message(content="Hello", role="assistant")
            choice = Choices(finish_reason="stop", index=0, message=msg)
            response = ModelResponse(
                id="chatcmpl-test",
                choices=[choice],
                model="gpt-4",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

            result = model_dump_with_preserved_fields(response)
            msg_dict = result["choices"][0]["message"]

            assert msg_dict["content"] == "Hello"
            assert "provider_specific_fields" not in msg_dict
        finally:
            litellm.flatten_provider_specific_fields = original
