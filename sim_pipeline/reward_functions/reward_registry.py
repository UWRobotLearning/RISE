from enum import Enum
from typing import Callable, Dict

MODULE_NAME_TO_FUNCTION: Dict[str, Dict[str, Callable]] = {}

def register_reward_func(func: Callable) -> Callable:
    """
    Decorator to register reward functions.
    """
    global MODULE_NAME_TO_FUNCTION
    module_name = func.__module__.split('.')[-1]
    func_name = func.__name__
    
    if module_name not in MODULE_NAME_TO_FUNCTION:
        MODULE_NAME_TO_FUNCTION[module_name] = {}
    
    MODULE_NAME_TO_FUNCTION[module_name][func_name] = func
    return func

def create_reward_function_enum() -> type:
    """
    Create and return the RewardFunction Enum class.
    """
    enum_dict = {}
    for module_name, funcs in MODULE_NAME_TO_FUNCTION.items():
        for func_name in funcs:
            enum_name = f"{module_name.upper()}_{func_name.upper()}"
            enum_value = f'{module_name}.{func_name}'
            enum_dict[enum_name] = enum_value
    
    return Enum('RewardFunction', enum_dict, type=RewardFunctionEnum)

class RewardFunctionEnum(Enum):
    def get_function(self) -> Callable:
        module_name, func_name = self.value.split('.')
        return MODULE_NAME_TO_FUNCTION[module_name][func_name]
