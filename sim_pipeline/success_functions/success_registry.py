from enum import Enum
from typing import Callable, Dict

SUCCESS_MODULE_NAME_TO_FUNCTION: Dict[str, Dict[str, Callable]] = {}

def register_success_func(func: Callable) -> Callable:
    """
    Decorator to register success functions.
    """
    global SUCCESS_MODULE_NAME_TO_FUNCTION
    module_name = func.__module__.split('.')[-1]
    func_name = func.__name__
    
    if module_name not in SUCCESS_MODULE_NAME_TO_FUNCTION:
        SUCCESS_MODULE_NAME_TO_FUNCTION[module_name] = {}
    
    SUCCESS_MODULE_NAME_TO_FUNCTION[module_name][func_name] = func
    return func

def create_success_function_enum() -> type:
    """
    Create and return the SuccessFunction Enum class.
    """
    enum_dict = {}
    for module_name, funcs in SUCCESS_MODULE_NAME_TO_FUNCTION.items():
        for func_name in funcs:
            enum_name = f"{module_name.upper()}_{func_name.upper()}"
            enum_value = f'{module_name}.{func_name}'
            enum_dict[enum_name] = enum_value
    
    return Enum('SuccessFunction', enum_dict, type=SuccessFunctionEnum)

class SuccessFunctionEnum(Enum):
    def get_function(self) -> Callable:
        module_name, func_name = self.value.split('.')
        return SUCCESS_MODULE_NAME_TO_FUNCTION[module_name][func_name]
