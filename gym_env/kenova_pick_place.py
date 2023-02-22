import numpy as np
import gym
import mujoco
from gym.envs.mujoco import mujoco_env, mujoco_rendering, 
from gym import utils
from gym.spaces import Box 
from typing import Optional, Union

Kenova_path = "mujoco_models\kenova_gen3_robotiq3f85\Kenova_2f85_pick_place.xml"

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    d = np.linalg.norm(goal_a - goal_b, axis=-1)
    return d

class MujocPickAndPlace(mujoco_env.MujocoEnv, utils.EzPickle)

    def __init__(
        self, 
        model_path=Kenova_path, 
        frame_skip=5,  
        reward_type='sparse', 
        target_in_the_air=True,
        render_mode=None):

        utils.EzPickle.__init__(
            self, 
            model_path, 
            frame_skip, 
            reward_type,
            target_in_the_air,
            render_mode
            )
        
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64)
        super.__init__(model_path, frame_skip, observation_space, render_mode)

        self.target_in_the_air = target_in_the_air
        self.reward_type = reward_type
        self.reward_type = reward_type
        self.distance_threshold = self.model.geom('object').size/2
        

        self._initialize_simulation()
        self._initialize_target_object_bondary()


        self.arm_home_qpos = self.init_qpos[:15]



    def _sample_object_target(self):
        object = np.array([
            np.random.uniform(*self.object_boundary['x']),
            np.random.uniform(*self.object_boundary['y']),
            self.object_boundary['z']
        ])
        while True:
            target = np.array([
                np.random.uniform(*self.object_boundary['x']),
                np.random.uniform(*self.object_boundary['y']),
                np.random.uniform(*self.object_boundary['z'])
            ])
            if np.linalg.norm(self.target - self.goal_pos) > 3 * self.distance_threshold:
                break
        return object, target

    
    def _initialize_target_object_bondary(self):
        table_size = self.model.geom('table').size
        table_pos = self.model.geom('table').pos
        object_size = self.model.geom('object').size
        target_size = self.model.geom('target0').size
        
        # object boundary is the table boundary 
        self.object_boundary = {
            'x':(0, table_size[0] + table_pos[0]),
            'y':(- table_size[1], table_size[0]),
            'z': object_size[2]/2
            }
            

        if self.target_in_the_air:
            z = (target_size[2]/2, 20 * target_size[2]) 
        else:
            z = (target_size[2]/2, ) * 2
        # target boundary is the table boundary, the z boundary is 20 times of its size
        self.target_boundary = {
            'x': self.object_boundary['x'],
            'y': self.object_boundary['y'],
            'z': z
            }
        

    def reset_model(self):
        qpos = self.init_qpos

        obj_pos, targ_pos = self._sample_object_target()

        # setting object and arm
        qpos[15:] = obj_pos + [0] * 4
        qpos[:15] = self.arm_home_qpos

        # setting target
        self.model.site('target0').pos = targ_pos
        qvel = self.init_qvel


        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """
        [7 arm joint, (7d)
        left follower, 
        right follower,
        xpos of gripper center: 1/2 * (left gripper + right gripper), (3d)
        xpos of object, (3d)
        xpos of target, (3d)
        """
        obs = np.concatenate([
            self.data.qpos[:7],
            self.data.qvel[:7],
            self.data.qpos[10], # left follower joint
            self.data.qpos[14], # right follower joint
            1/2 * (self.get_body_com('left_pad') + self.get_body_com('right_pad')),
            self.get_body_com('object0'),
            self.data.site('target0').xpos,
        ])
        return obs

    def compute_reward(self, gripper, object, target, action):
        grip_obj_dist = goal_distance(gripper   , object)
        obj_targ_dist = goal_distance(object, target)
        action_penalty = np.sum(np.square(action))
        is_success = ~(obj_targ_dist > self.distance_threshold)
        # sparse reward: return 0 when reach target, otherwise -1
        if self.reward_type == "sparse":
            reward =  (is_success).astype(np.float32) - 1
        # sparse reward: 
        else:
            reward = - obj_targ_dist - 0.5 * grip_obj_dist - 0.1 * action_penalty
        
        return reward, obj_targ_dist, grip_obj_dist, is_success
        
    def _set_action(self, action):
        assert self.data.ctrl.shape == action.shape, f'Value error: Action must have the same shape as ctrl' 
        # ctrl = self.data.ctrl + action
        self.do_simulation(action)


    def step(self, action:np.ndarray):
        ob = self._get_obs()
        gripper, obj, target = ob[16:19], ob[19:22], ob[22:]
        reward, object_target_dist, gripper_object_dist, is_success =\
            self.compute_reward(gripper, obj, target)        
        
        self._set_action(action, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        return (
            ob,
            reward,
            is_success,
            False,
            dict(object_target_dist=object_target_dist, gripper_object_dist=gripper_object_dist),
        )



    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

