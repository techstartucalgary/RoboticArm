from kenova_pick_place import KenovaPickAndPlace
from gym.wrappers.rescale_action import RescaleAction

if __name__ == "__main__":
    kpa = KenovaPickAndPlace()

    print(kpa.action_space)
    print(kpa.observation_space)

    kpa = RescaleAction(kpa, -1, 1)
    print(kpa.action_space)
    # obs = kpa.reset()
    # print(obs)
    action_space = kpa.action_space

    n_steps = 100
    for i in range(n_steps):
        action = kpa.action_space.sample()
        print(action)
        obs, reward, is_success, _, info = kpa.step(action)
        # kpa.render()
        print(obs, reward, is_success, _, info)
