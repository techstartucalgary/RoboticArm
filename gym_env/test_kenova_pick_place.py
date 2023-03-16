import numpy as np
from kenova_fetch.kenova_pick_place import KenovaPickAndPlace
from gym.wrappers.rescale_action import RescaleAction
import gym
import kenova_fetch

import sys,os
print( os.path.dirname(__file__))
if __name__ == "__main__":
    # kpa = KenovaPickAndPlace()
    kpa = gym.make('Kenova_pick_and_place-v0', render_mode = "human", reward_type = "dense")
    # kpa = gym.make('Kenova_slide-v0', render_mode = "human", reward_type = "dense")
    # kpa = gym.make('Kenova_push-v0', render_mode = "human", reward_type = "dense")
    # kpa = gym.make('Kenova_reach-v0', render_mode = "human", reward_type = "dense")

    print(kpa.action_space)
    print(kpa.observation_space)

    # kpa = RescaleAction(kpa, -1, 1)

    action_space = kpa.action_space

    # kpa.render_mode = "human"
    obs = kpa.reset()
    print(kpa.observation_space)
    print(obs)
    print(kpa.seed)
    # print(kpa.spec.id)
    action = kpa.action_space.sample()
    print(action)

    n_steps = 500
    # special_action = np.array([0, 1.16, -3.06, -0.825, -0.22, -0.81, 1.85, 1.47])
    for i in range(n_steps):
        if i % 10 == 0:
            print(kpa.reset())
        action = kpa.action_space.sample()
        print(action)
        # action = np.zeros(action_space.shape)
        # action[3] = 0.5
        # print(action)
        obs, reward, is_success, _, info = kpa.step(action)
        kpa.render()
        print(reward)

        # print(obs, reward, is_success, _, info)
