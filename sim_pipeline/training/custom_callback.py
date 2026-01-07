import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sim_pipeline.eval.rollout_env import PossibleVecEnv
from sim_pipeline.eval.rollout_policies.sb_rollout_policy import StableBaselinesRolloutPolicy
from sim_pipeline.eval.rollout import rollout
from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.eval.rollout_stats import RolloutStats

class LoggerCallback(BaseCallback):
    """
    A custom callback that derives from BaseCallback.
    """

    def __init__(
        self, eval_envs, eval_interval, save_dir, save_interval, verbose=False
    ):
        super(LoggerCallback, self).__init__(verbose)

        self.eval_interval = eval_interval

        self.save_dir = save_dir
        self.save_interval = save_interval

        self.eval_envs = PossibleVecEnv(eval_envs)
        self.eval_envs.reset()
        
        self.rollout_stats = RolloutStats()

        self.reward_sum = 0.0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        self.reward_sum += np.mean(self.locals["rewards"])

        if np.all(self.locals["dones"]):
            self.logger.record("eval/return", self.reward_sum)
            self.reward_sum = 0

        if (self.n_calls - 1) % self.eval_interval == 0:
            rollout_model = StableBaselinesRolloutPolicy(self.model)
            rollout(rollout_model, self.eval_envs, self.rollout_stats, self.logger, total_steps=200, tag="eval")

        if (self.n_calls - 1) % self.save_interval == 0:
            self.model.save(self.save_dir / "_step_" / str(self.n_calls))
            # self.model.save_replay_buffer(self.save_dir / "_replay_buffer.pkl")

        return True
