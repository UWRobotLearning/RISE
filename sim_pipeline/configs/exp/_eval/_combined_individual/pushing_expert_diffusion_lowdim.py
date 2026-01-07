from dataclasses import dataclass
from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig

@dataclass
class DiffusionEvalExpConfig(EvalExpConfig):
    policy_type: PolicyType = PolicyType.DIFFUSION 
    date_string: str = '0'
    