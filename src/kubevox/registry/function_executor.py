from typing import Any, Dict, Callable
import inspect
from loguru import logger


class FunctionExecutor:
    """Executes registered functions and handles their responses."""

    @staticmethod
    async def execute_function(func: Callable, **kwargs) -> Dict[str, Any]:
        """
        Execute a function with the given parameters and format its response.

        Args:
            func: The function to execute
            kwargs: Parameters to pass to the function

        Returns:
            Dict containing the execution results and formatted response
        """
        try:
            logger.info(f"Executing function: {func.__name__} with params: {kwargs}")

            # Check if function is async
            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)

            # Get response template from function metadata
            template = func.metadata.get("response_template", "")

            # Format the response if template exists
            formatted_response = template.format(**result) if template else str(result)

            logger.info(f"Function {func.__name__} executed successfully")

            return {
                "success": True,
                "result": result,
                "formatted_response": formatted_response,
            }

        except Exception as e:
            error_msg = f"Error executing {func.__name__}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
