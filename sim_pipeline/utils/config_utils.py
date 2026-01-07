from enum import Enum

def convert_hydra_for_serialization(hydra_cfg: dict):
    # convert enums into their values for json serialization
    
    for key, value in hydra_cfg.items():
        if isinstance(value, Enum):
            hydra_cfg[key] = value.value
        elif isinstance(value, dict):
            convert_hydra_for_serialization(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    convert_hydra_for_serialization(item)
                elif isinstance(item, Enum):
                    value[i] = item.value
