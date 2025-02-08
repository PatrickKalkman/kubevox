"""
Convert registered Kubernetes functions to Llama tools specification format.
Generate system prompts for Llama model interaction.
"""

import json
from typing import Any, Dict, List

START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"

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
            parameters = {"type": "dict", "required": [], "properties": {}}
        else:
            # Convert the existing parameters format to match Llama's format
            parameters = {
                "type": "dict",
                "required": parameters.get("required", []),
                "properties": parameters.get("properties", {}),
            }

        tool = {"name": func_name, "description": func_description, "parameters": parameters}
        tools.append(tool)

    return tools


def generate_system_prompt() -> str:
    """
    Generate the system prompt for Llama model including available functions.

    Returns:
        String containing the system prompt with embedded function definitions.
    """
    function_definitions = json.dumps(generate_llama_tools_schema(), indent=2)

    system_prompt = f"""{START_HEADER}system{END_HEADER}
You are an expert in composing functions. You are given a question and a set of possible functions. \
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. \
If none of the function can be used, point it out. If the given question lacks the parameters required
by the function, also point it out. You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of \
[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.

{function_definitions}
{EOT}"""

    return system_prompt


def generate_user_message(message: str) -> str:
    """
    Generate a formatted user message with the appropriate tokens for Llama model interaction.

    Args:
        message: The user's input message to be formatted.

    Returns:
        String containing the formatted user message with tokens.
    """
    return f"""{START_HEADER}user{END_HEADER}
{message}{EOT}"""


def generate_assistant_header() -> str:
    """
    Generate the assistant header token for Llama model interaction.

    Returns:
        String containing the assistant header token.
    """
    return f"{START_HEADER}assistant{END_HEADER}"
