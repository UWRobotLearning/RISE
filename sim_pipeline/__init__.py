import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', module='robosuite')
warnings.filterwarnings('ignore', module='robomimic')

# required for running on hyak without docker
if not os.environ.get("MUJOCO_GL"):
    try:
        from robosuite.utils.binding_utils import MjRenderContextOffscreen
    except ImportError:
        os.environ["MUJOCO_GL"] = "osmesa"
    else:
        os.environ["MUJOCO_GL"] = "glfw"
    
from sim_pipeline.configs.parse_config import parse_config

parse_config()

# print(f'Using renderer: {os.environ["MUJOCO_GL"]}')

from robosuite import macros

# if this is True, robosuite will try to use EGL rendering backend
# regardless of MUJCO_GL which errors. This forces glfw if enabled
# Can change this to True if EGL is supported.
macros.MUJOCO_GPU_RENDERING = False

# get rid of robosuite warning after first import
import robosuite
robosuite_path = Path(robosuite.__file__).parent
# make macros file
macros_path = robosuite_path / 'macros_private.py'
macros_path.touch()