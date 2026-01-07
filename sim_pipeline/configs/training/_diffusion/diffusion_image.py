from dataclasses import dataclass, field
from pathlib import Path
from sim_pipeline.configs.constants import CONFIG_DIR

@dataclass
class DiffusionImageTrainingConfig:
    num_epochs: int = 8000
    batch_size: int = 64
    
    default_config_path: str = str(CONFIG_DIR / 'diffusion_base_configs' / 'image_square_cnn.yaml')
    
    low_dim_keys: list[str] = field(default_factory=lambda: [
        "robot0_gripper_qpos",
        "robot0_eef_pos",
        "robot0_eef_quat",
    ])
    
    rgb_keys: list[str] = field(default_factory=lambda: [
        "agentview_image",
        "robot0_eye_in_hand_image",
    ])
    
    depth_keys: list[str] = field(default_factory=lambda: [
    ])
    
    execution_horizon: int = 8
    obs_horizon: int = 2
    prediction_horizon: int = 16
        
    render_video: bool = True
    rollout_horizon: int = 200