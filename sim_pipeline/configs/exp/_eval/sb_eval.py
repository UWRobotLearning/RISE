from dataclasses import dataclass
from sim_pipeline.configs.constants import PolicyType
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig

@dataclass
class RLEvalExpConfig(EvalExpConfig):
    # what type of policy to roll out
    policy_type: PolicyType = PolicyType.STABLE_BASELINES