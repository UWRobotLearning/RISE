from sim_pipeline.reward_functions.reward_registry import create_reward_function_enum

from sim_pipeline.reward_functions import square_reward

RewardFunction = create_reward_function_enum()

__all__ = ['RewardFunction']