from dataclasses import dataclass, field
from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig

@dataclass
class PDEvalExpConfig(EvalExpConfig):
    policy_type: PolicyType = PolicyType.PD 
    # default robomimic ee init
    target_pos: list[float] = field(default_factory=lambda: 
        [-0.1031, -0.0035, 1.015]
    )