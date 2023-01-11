import sys

import gym
import random
import numpy as np
import mujoco_py

total_episodes = 5
max_steps = 100

# env = gym.make("FetchReach-v1")
env = gym.make("FetchPickAndPlace-v1")
env.reset()
print(env.initial_state)
print(env.observation_space)
# print(env.observation_space.sample())

print(env.observation_space)
i = 0
gap = 1
while True:
    env.render()
    action = [0, 0, 0, gap]
    env.step(action)
    if i > 0:
        gap = gap * -1
    if i < -0:
        gap = gap * -1
    i += gap
    print(i, gap)

exit()
for episode in range(total_episodes):
    print("epsoide: {episode}")
    for steps in range(max_steps):
        env.render()
        action = env.action_space.sample()
        env.step(action)

env.close()
