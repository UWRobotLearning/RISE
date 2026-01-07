from dataclasses import dataclass, field
from pathlib import Path
from sim_pipeline.configs.constants import CONFIG_DIR
from sim_pipeline.configs.training._diffusion.base_diffusion import DiffusionTrainingConfig

@dataclass
class DiffusionTrainingRohanConfig(DiffusionTrainingConfig):
    num_epochs: int = 5000
    batch_size: int = 256
    
    default_config_path: str = str(CONFIG_DIR / 'diffusion_base_configs' / 'lowdim_square_cnn.yaml')
    
    low_dim_keys: list[str] = field(default_factory=lambda: [
        "object",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ])
    
    rgb_keys: list[str] = field(default_factory=lambda: [
    ])
    
    depth_keys: list[str] = field(default_factory=lambda: [
    ])
    
    execution_horizon: int = 8
    obs_horizon: int = 2
    prediction_horizon: int = 16
        
    render_video: bool = True
    rollout_horizon: int = 400