import numpy as np
from kenova_pick_place import KenovaPickAndPlace
from gym.wrappers.rescale_action import RescaleAction
import gym

if __name__ == "__main__":
    # kpa = KenovaPickAndPlace(render_mode=None)
    kpa = gym.make('Kenova_pick_and_place-v0')

    print(kpa.action_space)
    print(kpa.observation_space)

    kpa = RescaleAction(kpa, -1, 1)
    print(kpa.action_space)
    action_space = kpa.action_space
    kpa.reset()
    print(kpa.spec.id)

    n_steps = 100
    for i in range(n_steps):
        # if i % 100 == 0:
        # action = kpa.action_space.sample()
        action = np.zeros(action_space.shape)
        # print(action)
        obs, reward, is_success, _, info = kpa.step(action)
        kpa.render()

        print(obs, reward, is_success, _, info)
