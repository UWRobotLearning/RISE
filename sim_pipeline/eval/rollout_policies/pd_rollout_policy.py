import numpy as np
from scipy.spatial.transform import Rotation as R
from sim_pipeline.eval.rollout_policies.rollout_policy import RolloutPolicy

class PDRolloutPolicy(RolloutPolicy):
    def __init__(
            self,
            policy,
            target_pos: np.ndarray,
            action_chunking: bool = False,
            n_obs_steps: int | None = None,
        ):
        super().__init__(policy, action_chunking=action_chunking, n_obs_steps=n_obs_steps)
        
        self.target_pos = target_pos
        self.kp = 3
        self.kd = 0.1
        self.kpr = 0.1
        self.yaw_gain = 0.1

    def reset(self):
        self.prev_pos_error = np.zeros(3)
    
    def predict(self, obs, *args, **kwargs):
        # PD controller
        ee_position = obs['robot0_eef_pos']
        pos_error = self.target_pos - ee_position

        quat = obs['robot0_eef_quat']
        # switch from scalar first to scalar last
        quat = np.concatenate([quat[:, 1:], quat[:, :1]], axis=1)
        rot = R.from_quat(quat)
        yaw = rot.as_euler('xyz', degrees=False)[:, 0]
        orientation = rot.as_matrix().dot(np.array([1, 0, 0]))
        rot_error = np.cross(orientation, np.array([1, 0, 0]))

        # derivative term
        d_error = pos_error - self.prev_pos_error
        self.prev_pos_error = pos_error

        # PD control
        action_pos = self.kp * pos_error + self.kd * d_error 
        
        # rotation + yaw
        action_rot = -self.kpr * rot_error
        action_rot[:, 0] = -self.yaw_gain * yaw
        action_rot[:, [0, 1, 2]] = action_rot[:, [2, 1, 0]]        

        # no rotation
        action = np.concatenate([action_pos, action_rot], axis=1)
                
        # open gripper
        action = np.concatenate([action, np.zeros((action.shape[0], 1), dtype=action.dtype) - 1.0], axis=1)
        # clamp
        action = np.clip(action, -1, 1)
        
        return action