import numpy as np
import mujoco_env
from gym import utils
from gym.spaces import Box, Dict
from gym.utils import seeding
from typing import List, Union
import gym

Kenova_path = "Kenova_2f85_reach.xml"

MAX_CTRL_CHANGE = np.array([0.01, 0.01, 0.02, 0.02, 0.05, 0.05, 0.05, 10])
MAX_CTRL_CHANGE = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 10])

gym.envs.register(
    id='Reach_stage1',
    entry_point='kenova_fetch_fixed_reach:KenovaFetchFixedReach',
    max_episode_steps=500,
    kwargs = {'target_pos':[0.3, 0.4]}

)

gym.envs.register(
    id='Reach_stage2',
    entry_point='kenova_fetch_fixed_reach:KenovaFetchFixedReach',
    max_episode_steps=500,
    kwargs = {'target_pos':[0.3, -0.4]}
)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    d = np.linalg.norm(goal_a - goal_b, axis=-1)
    return d

MAX_CTRL_CHANGE = np.array([0.01, 0.01, 0.02, 0.02, 0.05, 0.05, 0.05, 10])

class KenovaFetchFixedReach(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_path=Kenova_path,
        frame_skip=1000,
        reward_type="sparse",
        target_in_the_air=False,
        render_mode="human",
        has_object = False,
        max_action_change:np.ndarray = MAX_CTRL_CHANGE,
        block_gripper = None,
        target_pos = [0.3, 0.4],
        init_arm_pos = "",
    ):
        utils.EzPickle.__init__(
            self, model_path, frame_skip, reward_type, target_in_the_air, render_mode
        )

        assert block_gripper in [None, 'closed', 'open'], \
            f'blocker_gripper can only be None, open or closed, get {block_gripper}'
        self.block_gripper = block_gripper
        self.max_action_change = max_action_change
        self.target_in_the_air = target_in_the_air
        self.reward_type = reward_type
        self.has_object = has_object
        self.init_target_pos = target_pos
        
        observation_space = Dict(
            dict(
                observation=Box(
                    -np.inf, np.inf, shape=(25,), dtype="float64"
                ),
                desired_goal=Box(
                    -np.inf, np.inf, shape=(3,), dtype="float64"
                ),
                achieved_goal=Box(
                    -np.inf, np.inf, shape=(3,), dtype="float64"
                ),
                gripper_center=Box(
                    -np.inf, np.inf, shape=(3,), dtype="float64"
                )
            )
        )
        mujoco_env.MujocoEnv.__init__(
            self, model_path, frame_skip, observation_space, render_mode
        )

        self.seed()

        # threshold is the radius of the target
        self.distance_threshold = self.model.site("target0").size[0]

        self._initialize_simulation()
        self._initialize_target_object_bondary()

        self.arm_home_qpos = self.data.qpos[:15]
        self.init_ctrl = None

        if init_arm_pos != "":
            initial_pos = np.loadtxt(init_arm_pos)
            assert initial_pos.shape == self.arm_home_qpos[:7].shape
            print('Read initial position: ', initial_pos)
            self.arm_home_qpos[:7] = initial_pos
            self.init_ctrl = initial_pos

    # override the function
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(float)
        bounds[:, 0] = -1.0
        bounds[:, 1] =  1.0

        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=float)
        assert self.action_space.shape == self.max_action_change.shape, \
            f"Incorrect max_action_change shape {self.max_action_change.shape} from action space shape {self.action_space.shape}"
        return self.action_space

    def _sample_object(self):
        object = np.array(
            [
                np.random.uniform(*self.object_boundary["x"]),
                np.random.uniform(*self.object_boundary["y"]),
                self.object_boundary["z"],
            ]
        )
        return object

    def _sample_target(self, reached):
        target = np.array([*self.init_target_pos, self.target_boundary['z'][0]])
        return target

    def _initialize_target_object_bondary(self):
        table_size = self.model.geom("table").size
        table_pos = self.model.geom("table").pos
        target_size = self.model.site("target0").size

        if self.has_object:
            try:
                object_size = self.model.geom("object0").size
            except KeyError as e:
                print(e, 'No object found, set has_object as False')
                self.has_object = False

            # object boundary is the table boundary
            self.object_boundary = {
                "x": (0 + table_size[0]/4, table_size[0] + table_pos[0]),
                "y": (-table_size[1], table_size[1]),
                "z": object_size[2] / 2,
            }

        if self.target_in_the_air:
            z = (target_size[2], 0.8)
        else:
            z = (target_size[2],) * 2
        # target boundary is the table boundary, the z boundary is 20 times of its size
        self.target_boundary = {
            "x": (-table_pos[0] + table_size[0]/4, table_size[0]),
            "y": (-table_size[1], table_size[1]),
            "z": z,
        }

    def reset_model(self):
        qpos = self.init_qpos.copy()

        if self.has_object:
            object_pos = self._sample_object()
            target_pos = self._sample_target(object_pos)
            # setting object 
            try:
                qpos[15:] = np.concatenate([object_pos, np.array([0] * 4)], axis=-1)
            except IndexError as e:
                print(e, 'No object found, set has_object as False')
                self.has_object = False
        else:
            gripper_center = self._get_obs()['gripper_center']
            target_pos = self._sample_target(gripper_center)


        # setting arm
        qpos[:15] = self.arm_home_qpos

        # setting target
        self.model.site("target0").pos[:] = target_pos
        qvel = self.init_qvel

        self.set_state(qpos, qvel)
        if self.init_ctrl is not None:
            ctrl = self.data.ctrl.copy()
            ctrl[:7] = self.init_ctrl
            self.do_simulation(ctrl, None)
        return self._get_obs()

    def _get_obs(self):
        """
        Dict(
            obs:(29,)
                7 arm joint pos, (7d)
                7 arm joint vel, (7d)
                left follower joint,  (1d)
                right follower joint, (1d)
                xpos of gripper center: 1/2 * (left gripper + right gripper), (3d)
                xpos of object, (3d)
                xpos of target, (3d)
            achieved_goal: (3d)
            desired_goal: (3d)
            )
        """
        if self.has_object:
            achieved_goal = self.get_body_com("object0")
        else:
            # if no object, set gripper center as achieved goal
            achieved_goal = 1/2 *(self.get_body_com("left_pad") + self.get_body_com("right_pad"))
        desired_goal = self.data.site("target0").xpos
        obs = np.concatenate(
            [
                self.data.qpos[:7],
                self.data.qvel[:7],
                self.data.qpos[10:11],  # left follower joint
                self.data.qpos[14:15],  # right follower joint
                1/2 *(self.get_body_com("left_pad") + self.get_body_com("right_pad")),
                achieved_goal,
                self.data.site("target0").xpos,
            ],
            axis=-1,
        )
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal' : desired_goal.copy(),
            'gripper_center': 1/2 *(self.get_body_com("left_pad") + self.get_body_com("right_pad")),
        }

    def compute_reward(self, reached_goal, desired_goal, gripper_center):
        d = goal_distance(desired_goal, reached_goal)
        is_success = d < self.distance_threshold
        if self.reward_type == "sparse":
            reward = (is_success).astype(float) - 1
        
        # dense reward:
        else:
            if self.has_object:
                grip_obj_dist = goal_distance(gripper_center, reached_goal)
                obj_targ_dist = d
                reward = obj_targ_dist + 0.5 * grip_obj_dist 
            else:
                reward = d 
            
            # reward = - np.sqrt(reward/3)
            #reward = - np.log(reward / 2) - 1.25
            reward = -reward

        return reward, is_success


    def _set_action(self, action):
        assert (
            self.data.ctrl.shape == action.shape
        ), f"Value error: Action must have the same shape as ctrl"
        # action is the change of the ctrl
        action_ = action.copy()
        action_ *= np.array(self.max_action_change)
        ctrl = self.data.ctrl + action_
        self.do_simulation(ctrl, None)

    def step(self, action: np.ndarray):
        ob = self._get_obs()
        gripper, reached_goal, desired_goal =\
             ob['gripper_center'], ob['achieved_goal'], ob['desired_goal']
        reward, is_success = self.compute_reward(reached_goal, desired_goal, gripper)
        
        if is_success:
            print("Sucess, the arm qposition is: ", self.data.qpos[:15])
        # add action penalty
        if self.reward_type:
            reward -= 0.1* np.linalg.norm(action)
        info = dict(is_success=is_success)

        self._set_action(action)
        if self.render_mode == "human":
            self.render()

        return (
            ob,
            reward,
            is_success,
            False,
            info
        )

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
