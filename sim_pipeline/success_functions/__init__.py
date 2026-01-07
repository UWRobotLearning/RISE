from sim_pipeline.success_functions.success_registry import create_success_function_enum

from sim_pipeline.success_functions import square_success

SuccessFunction = create_success_function_enum()

__all__ = ['SuccessFunction']