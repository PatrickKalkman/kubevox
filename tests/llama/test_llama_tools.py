"""
Tests for the Llama tools schema generation functionality.
"""

import pytest

from kubevox.llama.llama_tools import (
    generate_assistant_header,
    generate_llama_tools_schema,
    generate_system_prompt,
    generate_user_message,
)
from kubevox.registry.function_registry import FunctionRegistry


@pytest.fixture
def sample_function():
    """Create a sample function with metadata for testing."""

    def test_func():
        pass

    test_func.metadata = {
        "description": "Test function description",
        "response_template": "Test response",
        "parameters": {
            "type": "object",
            "properties": {"test_param": {"type": "string", "description": "A test parameter"}},
            "required": ["test_param"],
        },
    }
    return test_func


def test_generate_llama_tools_schema(sample_function):
    """Test the generation of Llama tools schema from registered functions."""
    # Register the test function
    FunctionRegistry.functions = [sample_function]

    # Generate the schema
    schema = generate_llama_tools_schema()

    # Verify the schema structure
    assert isinstance(schema, list)
    assert len(schema) == 1

    tool = schema[0]
    assert tool["name"] == "test_func"
    assert tool["description"] == "Test function description"
    assert tool["parameters"]["type"] == "dict"
    assert tool["parameters"]["required"] == ["test_param"]
    assert "test_param" in tool["parameters"]["properties"]


def test_generate_llama_tools_schema_empty():
    """Test schema generation with no registered functions."""
    FunctionRegistry.functions = []
    schema = generate_llama_tools_schema()
    assert isinstance(schema, list)
    assert len(schema) == 0


def test_generate_llama_tools_schema_no_parameters(sample_function):
    """Test schema generation for a function without parameters."""
    # Remove parameters from the test function
    sample_function.metadata.pop("parameters")
    FunctionRegistry.functions = [sample_function]

    schema = generate_llama_tools_schema()
    assert len(schema) == 1

    tool = schema[0]
    assert tool["parameters"] == {"type": "dict", "required": [], "properties": {}}


def test_generate_system_prompt(sample_function):
    """Test generation of system prompt with function definitions."""
    FunctionRegistry.functions = [sample_function]

    system_prompt = generate_system_prompt()

    # Check for required components
    assert system_prompt.startswith("<|start_header_id|>system<|end_header_id|>")
    assert system_prompt.endswith("<|eot_id|>")
    assert "You are an expert in composing functions" in system_prompt
    assert "test_func" in system_prompt  # Should contain our sample function
    assert "Test function description" in system_prompt


def test_generate_user_message():
    """Test generation of formatted user messages."""
    test_message = "List all pods in namespace default"
    formatted_message = generate_user_message(test_message)

    assert formatted_message.startswith("<|start_header_id|>user<|end_header_id|>")
    assert test_message in formatted_message
    assert formatted_message.endswith("<|eot_id|>")


def test_generate_assistant_header():
    """Test generation of assistant header."""
    header = generate_assistant_header()

    assert header == "<|start_header_id|>assistant<|end_header_id|>"
