from dataclasses import dataclass
from sim_pipeline.configs.exp._eval.robomimic_eval import RobomimicEvalExpConfig

@dataclass
class RobomimicEvalSaveDataExpConfig(RobomimicEvalExpConfig):
    save_to_dataset: bool = True
    verbose: bool = False