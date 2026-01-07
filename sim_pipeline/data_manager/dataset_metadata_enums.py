from enum import Enum

class ProcessedStatus(Enum):
    RAW_TELEOP = 'raw_teleop'
    PRE_PROCESSED = 'pre_processed'
    PROCESSED = 'processed'

class EnvType(Enum):
    ROBOMIMIC = 'robomimic'
    ROBOSUITE = 'robosuite'
    REAL_FRANKA = 'real_franka'
    REAL_TYCHO = 'real_tycho'

class DatasetType(Enum):
    PH = 'ph'
    MH = 'mh'
    RL = 'rl'
    PLAY = 'play'

class ObsType(Enum):
    IMAGE = 'image'
    STATE = 'low_dim'
    IMAGE_ONLY = 'image_only'
    
    def included_keys(self):
        if self == ObsType.IMAGE:
            return ['agentview_image', 'robot0_eye_in_hand_image']
        elif self == ObsType.STATE:
            return None
        elif self == ObsType.IMAGE_ONLY:
            return ['agentview_image', 'robot0_eye_in_hand_image']
        else:
            return None
        
    def excluded_keys(self):
        if self == ObsType.IMAGE:
            return None
        elif self == ObsType.STATE:
            return ['agentview_image', 'robot0_eye_in_hand_image']
        elif self == ObsType.IMAGE_ONLY:
            return None
        else:
            return None
        
    def includes_image_obs(self):
        return self in [ObsType.IMAGE, ObsType.IMAGE_ONLY]
    
    def includes_depth_obs(self):
        return False

class ActionType(Enum):
    RELATIVE = 'relative'
    ABSOLUTE = 'absolute'

class RewardType(Enum):
    SPARSE = 'sparse'
    DENSE = 'dense'
    NO_REWARD = 'no_reward'
    DENSITY_REWARD = 'density_reward'