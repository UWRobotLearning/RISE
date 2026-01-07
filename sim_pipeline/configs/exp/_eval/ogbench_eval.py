from dataclasses import dataclass
from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig

@dataclass
class OGBenchEvalExpConfig(EvalExpConfig):
    policy_type: PolicyType = PolicyType.OGBENCH 
    