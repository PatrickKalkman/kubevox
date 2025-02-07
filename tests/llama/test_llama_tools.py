"""
Tests for the Llama tools schema generation functionality.
"""

import pytest
from kubevox.llama.llama_tools import generate_llama_tools_schema
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
            "properties": {
                "test_param": {
                    "type": "string",
                    "description": "A test parameter"
                }
            },
            "required": ["test_param"]
        }
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
    assert tool["parameters"] == {
        "type": "dict",
        "required": [],
        "properties": {}
    }
