from dataclasses import dataclass
from sim_pipeline.configs.data.base_data import BaseDataConfig

@dataclass
class DataSummaryConfig:
    data: BaseDataConfig = BaseDataConfig()