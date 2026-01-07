from dataclasses import dataclass, field
from sim_pipeline.configs.constants import PolicyType, CombinedPolicyType
from sim_pipeline.configs.exp._eval.evaluation import EvalExpConfig

@dataclass
class HeuristicResetSplitEvalExpConfig(EvalExpConfig):
    policy_type: PolicyType = PolicyType.COMPOSITE 
    policy_configs: list[str] = field(default_factory=lambda: [
        'square_iql_diffusion_push_coverage_to_tl',
        'eval_pd',
        'square_expert_lowdim_rohan',
    ])
    combined_policy_type: CombinedPolicyType = CombinedPolicyType.HEURISTIC_RESET_SPLIT
    