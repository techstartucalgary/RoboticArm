import numpy as np
import gym
import mujoco

# from gym.envs.mujoco import mujoco_env
import mujoco_env
from gym import utils
from gym.spaces import Box, Dict
from gym.utils import seeding

Kenova_path = "Kenova_2f85_pick_place.xml"


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    d = np.linalg.norm(goal_a - goal_b, axis=-1)
    return d

gym.envs.register(
    id='Kenova_pick_and_place-v0',
    entry_point='kenova_pick_place:KenovaPickAndPlace',
    max_episode_steps=200,
)

class KenovaPickAndPlace(mujoco_env.MujocoEnv, utils.EzPickle):
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
        target_in_the_air=True,
        render_mode="human",
    ):
        utils.EzPickle.__init__(
            self, model_path, frame_skip, reward_type, target_in_the_air, render_mode
        )

        
        observation_space = Dict(
            dict(
                observation=Box(
                    -np.inf, np.inf, shape=(29,), dtype="float64"
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


        self.target_in_the_air = target_in_the_air
        self.reward_type = reward_type

        # threshold is the radius of the target
        self.distance_threshold = self.model.geom("object0").size[0] / 2

        self._initialize_simulation()
        self._initialize_target_object_bondary()

        self.arm_home_qpos = self.init_qpos[:15]



    def _sample_object_target(self):
        object = np.array(
            [
                np.random.uniform(*self.object_boundary["x"]),
                np.random.uniform(*self.object_boundary["y"]),
                self.object_boundary["z"],
            ]
        )
        while True:
            target = np.array(
                [
                    np.random.uniform(*self.target_boundary["x"]),
                    np.random.uniform(*self.target_boundary["y"]),
                    np.random.uniform(*self.target_boundary["z"]),
                ]
            )
            if np.linalg.norm(target - object) > 3 * self.distance_threshold:
                break
        return object, target

    def _initialize_target_object_bondary(self):
        table_size = self.model.geom("table").size
        table_pos = self.model.geom("table").pos
        object_size = self.model.geom("object0").size
        target_size = self.model.site("target0").size

        # object boundary is the table boundary
        self.object_boundary = {
            "x": (0, table_size[0] + table_pos[0]),
            "y": (-table_size[1], table_size[0]),
            "z": object_size[2] / 2,
        }

        if self.target_in_the_air:
            z = (target_size[2], 20 * target_size[2])
        else:
            z = (target_size[2],) * 2
        # target boundary is the table boundary, the z boundary is 20 times of its size
        self.target_boundary = {
            "x": (-table_pos[0], table_size[0]),
            "y": self.object_boundary["y"],
            "z": z,
        }

    def reset_model(self):
        qpos = self.init_qpos.copy()

        self.object_pos, self.target_pos = self._sample_object_target()

        # setting object and arm
        qpos[15:] = np.concatenate([self.object_pos, np.array([0] * 4)], axis=-1)
        qpos[:15] = self.arm_home_qpos

        # setting target
        self.model.site("target0").pos[:] = self.target_pos
        qvel = self.init_qvel

        self.set_state(qpos, qvel)
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
                rotational position of object (4d)
                xpos of target, (3d)
            achieved_goal: (3d)
            desired_goal: (3d)
            )
        """
        obs = np.concatenate(
            [
                self.data.qpos[:7],
                self.data.qvel[:7],
                self.data.qpos[10:11],  # left follower joint
                self.data.qpos[14:15],  # right follower joint
                1/2 *(self.get_body_com("left_pad") + self.get_body_com("right_pad")),
                self.get_body_com("object0"),
                self.data.qpos[-4:],
                self.data.site("target0").xpos,
            ],
            axis=-1,
        )
        achieved_goal = self.get_body_com("object0")
        desired_goal = self.data.site("target0").xpos
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal' : desired_goal.copy(),
            'gripper_center': 1/2 *(self.get_body_com("left_pad") + self.get_body_com("right_pad")),
        }

    def compute_reward(self, object, target, gripper,  action=None):
        obj_targ_dist = goal_distance(object, target)
        is_success = obj_targ_dist < self.distance_threshold
        # sparse reward: return 0 when reach target, otherwise -1
        if self.reward_type == "sparse":
            reward = (is_success).astype(np.float32) - 1
            return reward, obj_targ_dist, is_success
        # sparse reward:
        else:
            grip_obj_dist = goal_distance(gripper, object)
            action_penalty = np.sum(np.square(action))
            reward = -obj_targ_dist - 0.5 * grip_obj_dist - 0.1 * action_penalty

            return reward, obj_targ_dist, grip_obj_dist, is_success

    def _set_action(self, action):
        assert (
            self.data.ctrl.shape == action.shape
        ), f"Value error: Action must have the same shape as ctrl"
        # ctrl = self.data.ctrl + action
        self.do_simulation(action, None)

    def step(self, action: np.ndarray):
        ob = self._get_obs()
        gripper, obj, target = ob['gripper_center'], ob['achieved_goal'], ob['desired_goal']
        if self.reward_type == 'sparse':
            (   reward,
                object_target_dist,
                is_success,
            ) = self.compute_reward(gripper, obj, target, action)
            info =  dict(
                object_target_dist=object_target_dist,
                is_success=is_success
            )
        else:
            (
                reward,
                object_target_dist,
                gripper_object_dist,
                is_success,
            ) = self.compute_reward(gripper, obj, target, action)
            info =  dict(
                object_target_dist=object_target_dist,
                gripper_object_dist=gripper_object_dist,
                is_success=is_success
            )

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
