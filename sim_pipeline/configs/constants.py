import numpy as np

from enum import Enum
from pathlib import Path

import mimicgen_envs

class EnvType(Enum):
    FRANKA = 'franka'
    POINT_MAZE = 'point_maze'
    ROBOMIMIC = 'robomimic'

class RLAlgorithm(Enum):
    PPO = "PPO"
    SAC = "SAC"

    def get_algo(self):
        from stable_baselines3 import PPO, SAC

        if self == RLAlgorithm.PPO:
            return PPO
        elif self == RLAlgorithm.SAC:
            return SAC
        else:
            raise ValueError(f"Unknown algorithm {self}")
        
class ImitationAlgorithm(Enum):
    BC_RNN = 'bc_rnn'
    BC = 'bc'
    CQL = 'cql'
    IQL = 'iql'
    IDQL = 'idql'
    IDQL_QSM = 'idql_qsm'
    IQL_DIFFUSION = 'iql_diffusion'
    DISCRIMINATOR = 'discriminator'
    
    def is_offline_rl(self):
        return self in [ImitationAlgorithm.IQL, ImitationAlgorithm.CQL, ImitationAlgorithm.IQL_DIFFUSION, ImitationAlgorithm.IDQL_QSM, ImitationAlgorithm.IDQL]
    
    def is_diffusion(self):
        return self in [ImitationAlgorithm.IQL_DIFFUSION, ImitationAlgorithm.IDQL, ImitationAlgorithm.IDQL_QSM]
    
class PolicyType(Enum):
    ROBOMIMIC = 'robomimic'
    STABLE_BASELINES = 'sb3'
    DIFFUSION = 'diffusion'
    OGBENCH = 'ogbench'
    COMPOSITE = 'composite'
    PD = 'pd'
    DISCRIMINATOR = 'discriminator'
    QTRANSFORMER = 'qtransformer'
    
class OGBenchAlgo(Enum):
    GCBC = 'gcbc'
    GCIQL = 'gciql'
    HIQL = 'hiql'
    CRL = 'crl'
    
    def get_agent(self):
        from impls.agents import agents
        
        return agents[self.value]
        
class CombinedPolicyType(Enum):
    HEURISTIC_RESET = 'heuristic_reset'
    HEURISTIC_RESET_SPLIT = 'heuristic_reset_split'
class RobomimicEnvType(Enum):
    LIFT = 'Lift'
    STACK = 'Stack'
    NUT_ASSEMBLY = 'NutAssembly'
    NUT_ASSEMBLY_SQUARE = 'NutAssemblySquare'
    NUT_ASSEMBLY_ROUND = 'NutAssemblyRound'
    NUT_ASSEMBLY_HANG_INSERT = 'NutAssemblyHangInsert'
    PICK_PLACE = 'PickPlace'
    DOOR = 'Door'
    THREE_PIECE_ASSEMBLY = 'ThreePieceAssembly'
    THREADING = 'Threading'
    MUG_CLEANUP = 'MugCleanup'
    COFFEE = 'Coffee'
    TOOL_HANG = 'ToolHang'
    
    def get_object_names(self):
        # see object names in robosuite environments, i.e. robosuite/environments/manipulation/nut_assembly.py:403
        if self in [RobomimicEnvType.NUT_ASSEMBLY, RobomimicEnvType.NUT_ASSEMBLY_SQUARE, RobomimicEnvType.NUT_ASSEMBLY_ROUND]:
            return ['SquareNut', 'RoundNut']
        elif self == RobomimicEnvType.THREE_PIECE_ASSEMBLY:
            return ['Base', 'Piece1']
        elif self == RobomimicEnvType.THREADING:
            return ['Needle', 'Tripod']
        elif self == RobomimicEnvType.TOOL_HANG:
            return ['standObject','frameObject','toolObject']
        else:
            return []
        
    def get_default_obj_init_ranges(self):
        if self in [RobomimicEnvType.NUT_ASSEMBLY, RobomimicEnvType.NUT_ASSEMBLY_SQUARE, RobomimicEnvType.NUT_ASSEMBLY_ROUND]:
            return {
                'SquareNut':
                [
                    [-0.115, -0.11], 
                    [0.11, 0.225],
                ],
                'RoundNut':
                [
                    [-1.0, -0.9],
                    # [-0.115, -0.11], 
                    [-0.225, -0.11],
                ]    
            }
        elif self in [RobomimicEnvType.THREE_PIECE_ASSEMBLY]:
            return {
                'Base':
                [
                    [0.0, 0.0], 
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                'Piece1':
                [
                    [-0.22, 0.22], 
                    [-0.22, 0.22],
                    [1.5708, 1.5708],
                ],
                # 'Piece2':
                # [
                #     [-0.22, 0.22], 
                #     [-0.22, 0.22],
                #     [1.5708, 1.5708],
                # ]
            }
        elif self == RobomimicEnvType.THREADING:
            return {
                'Needle':
                [
                    [-0.2, -0.05], 
                    [0.15, 0.25],
                    [-2. * np.pi / 3., -np.pi / 3.],
                ],
                'Tripod':
                [
                    [0.0, 0.0], 
                    [-0.15, -0.15],
                    [np.pi / 2., np.pi / 2.],
                ]
            }
        elif self == RobomimicEnvType.TOOL_HANG:
            return {
                'standObject':
                [
                    [-0.08, -0.08], 
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                'frameObject':
                [
                    [-0.02, -0.06], 
                    [-0.26, -0.22],
                    [(-np.pi / 2) + (np.pi / 6) - np.pi / 18, (-np.pi / 2) + (np.pi / 6) + np.pi / 18],
                ],
                'toolObject':
                [
                    [-0.02, -0.06], 
                    [-0.22, -0.18],
                    [(-np.pi / 2) - (np.pi / 9.0) - np.pi / 18, (-np.pi / 2) - (np.pi / 9.0) + np.pi / 18],
                ],
            }
        else:
            return {}
        
    def get_simple_name(self):
        if self == RobomimicEnvType.NUT_ASSEMBLY_SQUARE:
            return 'square'
        else:
            return self.value.lower()
        
    def get_surface_ref_pos(self):
        # reference pos to pass into Placement Sampler
        if self in [RobomimicEnvType.NUT_ASSEMBLY, RobomimicEnvType.NUT_ASSEMBLY_SQUARE, RobomimicEnvType.NUT_ASSEMBLY_ROUND]:
            return np.array([0.0, 0.0, 0.82])
        elif self in [RobomimicEnvType.THREE_PIECE_ASSEMBLY, RobomimicEnvType.THREADING]:
            return np.array([0.0, 0.0, 0.8])
        elif self == RobomimicEnvType.TOOL_HANG:
            return np.array([0.0, 0.0, 0.8])
        else:
            return []

class IO_Devices(Enum):
    KEYBOARD = 'keyboard'
    SPACEMOUSE = 'spacemouse'
    SPAECMOUSE_ORIGINAL = 'spacemouse_original'

    def get_device(self, pos_sensitivity=1.0, rot_sensitivity=1.0):
        if self == IO_Devices.KEYBOARD:
            from sim_pipeline.data_collection.keyboard import Keyboard

            device = Keyboard(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        elif self == IO_Devices.SPACEMOUSE:
            from sim_pipeline.data_collection.spacemouse_hybrid import SpaceMouse as SpaceMouseHybrid

            device = SpaceMouseHybrid(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        elif self == IO_Devices.SPAECMOUSE_ORIGINAL:
            from robosuite.devices import SpaceMouse

            device = SpaceMouse(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity)
        else:
            raise ValueError(f"Unknown device {self}")
        
        return device

import sim_pipeline
ROOT_DIR = Path(sim_pipeline.__file__).parent
CONFIG_DIR = ROOT_DIR / 'configs'
LOG_DIR = ROOT_DIR / 'logs'
DATA_DIR = ROOT_DIR / 'data'
ROLLOUT_DATA_DIR = DATA_DIR / 'rollout_generated_data'
COMBINED_DATA_DIR = DATA_DIR / 'combined'
DATASET_DB_PATH = ROOT_DIR / 'datasets.db'
MODEL_DB_PATH = ROOT_DIR / 'models.db'

CALIBRATION_PATH = (Path(__file__).parent.parent.parent.parent / 'weird_franka' / 'weird_franka' / 'perception' / 'cameras' / 'calibration' /
                    'logs' / 'default.json')