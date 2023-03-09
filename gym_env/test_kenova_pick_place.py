import numpy as np
from kenova_pick_place import KenovaPickAndPlace
from gym.wrappers.rescale_action import RescaleAction
import gym

if __name__ == "__main__":
    # kpa = KenovaPickAndPlace(render_mode=None)
    kpa = gym.make('Kenova_pick_and_place-v0', render_mode = "human")

    print(kpa.action_space)
    print(kpa.observation_space)

    kpa = RescaleAction(kpa, -1, 1)

    action_space = kpa.action_space

    # kpa.render_mode = "human"
    obs = kpa.reset()
    print(kpa.observation_space)
    print(obs)
    print(kpa.seed)
    # print(kpa.spec.id)

    n_steps = 50
    for i in range(n_steps):
        # action = kpa.action_space.sample()
        action = np.zeros(action_space.shape)
        action[3] = 0.5
        # print(action)
        obs, reward, is_success, _, info = kpa.step(action)
        kpa.render()

        # print(obs, reward, is_success, _, info)
