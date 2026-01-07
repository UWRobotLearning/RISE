from dataclasses import dataclass, field
from sim_pipeline.configs.env.robomimic_square_ood import RobommimicSquareEnvOODConfig
from sim_pipeline.reward_functions import RewardFunction
from sim_pipeline.success_functions import SuccessFunction

@dataclass
class RobommimicSquareEnvOODPushingConfig(RobommimicSquareEnvOODConfig):
    horizon = 300

    reward_function: RewardFunction | None = RewardFunction.SQUARE_REWARD_REACH_AND_PUSH_REWARD_MID #RewardFunction.SQUARE_REWARD_REACH_AND_PUSH_REWARD
    success_function: SuccessFunction | None = SuccessFunction.SQUARE_SUCCESS_PUSHING_SUCCESS
