from dataclasses import dataclass, field
from sim_pipeline.configs.constants import ImitationAlgorithm

@dataclass
class ImitationTrainingConfig:
    algo: ImitationAlgorithm = ImitationAlgorithm.BC_RNN

    resume: bool = True
    
    low_dim_keys: list[str] = field(default_factory=lambda: [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object"
    ])
    
    rgb_keys: list[str] = field(default_factory=lambda: [
    ])
    
    depth_keys: list[str] = field(default_factory=lambda: [
    ])
    
    # only for robomimic, either 'low_dim', 'all' or None
    cache_mode: str | None = 'low_dim'
    batch_size: int = 100
    num_epochs: int = 2000
    epoch_every_n_steps: int = 100
    
    render_video: bool = True
    rollout_horizon: int = 200
    
    pad_seq_length: bool = True