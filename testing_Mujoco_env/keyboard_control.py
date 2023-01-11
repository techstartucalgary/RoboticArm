import pygame
import gym
import random
import numpy as np
# import keyboard

pygame.init()
pygame.display.set_mode((100, 100))


def key_listener():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                return 1
            elif event.key == pygame.K_RIGHT:
                return -1
            elif event.key == pygame.K_UP:
                return 2
            elif event.key == pygame.K_DOWN:
                return -2
            elif event.key == pygame.K_w:
                return 3
            elif event.key == pygame.K_s:
                return -3
            elif event.key == pygame.K_SPACE:
                return 0
            else:
                return None


if __name__ == '__main__':
    env = gym.make("FetchPickAndPlace-v1")
    env.reset()
    print(env.observation_space)
    # exit()
    Vx, Vy, Vz, F = 0, 0, 0, -1
    counter = 0
    while True:
        env.render()
        key = key_listener()
        base_speed = 0.8
        if key is not None:
            if abs(key) == 1:
                Vy = -1 * base_speed * abs(key) / key
            if abs(key) == 2:
                Vx = -1 * base_speed * abs(key) / key
            if abs(key) == 3:
                Vz = base_speed * abs(key) / key
            if key == 0:
                F = 1 if F == -1 else -1
        else:
            Vx, Vy, Vz = 0, 0, 0

        action = [Vx, Vy, Vz, F]
        obs, reward, done, info = env.step(action)
        print(obs['observation'])
        # print(done, reward, counter)
        if reward > -1:
            print(obs)
            env.reset()
