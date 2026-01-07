import math
import numpy as np

from sim_pipeline.configs.exp._eval.diffusion_eval import DiffusionEvalExpConfig
from sim_pipeline.eval.eval import setup_eval
from sim_pipeline.eval.rollout import rollout
from sim_pipeline.eval.rollout_stats import RolloutStats

class DiffusionEnvRunner:
    def __init__(self, config: DiffusionEvalExpConfig, n_envs: int, n_train: int, n_test: int):
        self.config = config
        self.n_envs = n_envs
        self.n_inits = n_train + n_test
        self.env, self.policy, self.render_mode, video_writers = setup_eval(config)
    
    def run(self, policy):
        n_chunks = math.ceil(self.n_inits / self.n_envs)
        rollout_stats = RolloutStats()
        
        for chunk_idx in range(n_chunks):
            rollout(
                policy=policy,
                eval_envs=self.env,
                rollout_stats=rollout_stats,
                logger=None,
                total_steps=self.config.rollout_horizon,
                render_mode=self.render_mode,
                log_video=False,
                video_writers=None,
                render_dims=None,
                return_trajectory=False,
                end_on_success=False,
            )
            
        rollout_stats.compute_statistics()
        
        return rollout_stats.get_avg_stats()