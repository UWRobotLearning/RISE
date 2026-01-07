from typing import TypeVar

T = TypeVar('T')

def get_base_env(env, desired_class: type[T]) -> T | None:
    if isinstance(env, desired_class):
        return env
    
    if not hasattr(env, 'env'):
        return None
    
    return get_base_env(env.env, desired_class)