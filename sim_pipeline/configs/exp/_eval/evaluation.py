from dataclasses import dataclass, field
from omegaconf import MISSING
from sim_pipeline.configs.exp.base import BaseExpConfig
from sim_pipeline.configs.constants import PolicyType

@dataclass
class EvalExpConfig(BaseExpConfig):
    # seed eval
    eval_seed: int = MISSING
    
    model_path: str | None = None
    
    # what type of policy to roll out
    policy_type: PolicyType = PolicyType.ROBOMIMIC
    
    # whether to save video of evaluation rollouts
    log_video: bool = True
    # dimensions to render video
    video_dims: tuple | None = field(default_factory=lambda: [196,196])
    # only record every nth frame to save space
    video_skip: int = 5
    
    num_rollouts: int = 20
    rollout_horizon: int = 400
    
    # whether to save trajectories to a new dataset
    save_to_dataset: bool = False
    # if saving to dataset, name of file to save to
    dataset_name: str = 'eval_default'
    
    # if > 1, use vectorized env with this many parallel envs
    num_workers: int = 1
    
    # print out stats every episode
    verbose: bool = True
    
    # whether to use the model manager. Otherwise, specify
    # the identifier (i.e. the date string for diffusion/robomimic)
    use_model_manager: bool = True
    
    # whether to reset to the exact intial states in the dataset
    reset_to_data: bool = False
    
    manual_lookahead: bool = False
    opex: bool = False
    
    discriminator_name: str | None = None
    use_discriminator: bool = False