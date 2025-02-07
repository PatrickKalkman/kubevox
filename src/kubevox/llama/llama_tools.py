"""
Convert registered Kubernetes functions to Llama tools specification format.
"""

from typing import List, Dict, Any
from kubevox.registry.function_registry import FunctionRegistry

def generate_llama_tools_schema() -> List[Dict[str, Any]]:
    """
    Convert all registered functions to Llama tools specification format.
    
    Returns:
        List of dictionaries containing function definitions in Llama tools format.
    """
    tools = []
    for func in FunctionRegistry.functions:
        func_name = func.__name__
        func_description = func.metadata.get("description", "")
        parameters = func.metadata.get("parameters")

        if parameters is None:
            parameters = {
                "type": "dict",
                "required": [],
                "properties": {}
            }
        else:
            # Convert the existing parameters format to match Llama's format
            parameters = {
                "type": "dict",
                "required": parameters.get("required", []),
                "properties": parameters.get("properties", {})
            }

        tool = {
            "name": func_name,
            "description": func_description,
            "parameters": parameters
        }
        tools.append(tool)

    return tools
