import json
from typing import Any, Callable, Dict, Literal, Optional, get_type_hints


class FunctionRegistry:
    functions = []

    @classmethod
    def register(
        cls,
        description: str,
        response_template: str,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """Decorator to register a function with the registry."""

        def decorator(func: Callable):
            # Attach metadata to the function
            func.metadata = {
                "description": description,
                "response_template": response_template,
                "parameters": parameters,
            }
            cls.functions.append(func)
            return func

        return decorator

    @classmethod
    def generate_json_schema(cls) -> str:
        """Generates a JSON schema string from all registered functions."""
        functions_schema = []
        for func in cls.functions:
            func_name = func.__name__
            func_description = func.metadata.get("description", "")
            parameters = func.metadata.get("parameters")

            if parameters is None:
                # Infer parameters from type hints
                type_hints = get_type_hints(func)
                parameters = {"type": "object", "properties": {}, "required": []}
                for param, param_type in type_hints.items():
                    if param == "return":
                        continue
                    param_schema = {}
                    if getattr(param_type, "__origin__", None) is Literal:
                        param_schema["type"] = "string"
                        param_schema["enum"] = list(param_type.__args__)
                    else:
                        param_schema["type"] = param_type.__name__
                    parameters["properties"][param] = param_schema
                    parameters["required"].append(param)

            function_schema = {
                "type": "function",
                "name": func_name,
                "description": func_description,
                "parameters": parameters,
            }
            functions_schema.append(function_schema)

        return json.dumps(functions_schema)
