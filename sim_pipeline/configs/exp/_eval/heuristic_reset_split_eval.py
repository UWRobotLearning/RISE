from dataclasses import dataclass, field
from sim_pipeline.configs.constants import PolicyType, CombinedPolicyType
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig

@dataclass
class HeuristicResetSplitEvalExpConfig(EvalExpConfig):
    policy_type: PolicyType = PolicyType.COMPOSITE 
    policy_configs: list[str] = field(default_factory=lambda: [
        'pushing_expert_lowdim',
        'eval_pd',
        'square_expert_lowdim',
    ])
    combined_policy_type: CombinedPolicyType = CombinedPolicyType.HEURISTIC_RESET
    