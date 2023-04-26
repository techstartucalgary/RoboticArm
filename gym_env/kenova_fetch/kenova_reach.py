import gym
import os, sys
base_file = os.path.join(os.path.dirname(__file__),'..')
sys.path.append(base_file)
from kenova_fetch_base import KenovaFetchBase

Kenova_path = "Kenova_2f85_reach.xml"
import numpy as np

MAX_CTRL_CHANGE = np.array([0.01, 0.01, 0.02, 0.02, 0.05, 0.05, 0.05, 10])
MAX_CTRL_CHANGE = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 10])

gym.envs.register(

    id='Kenova_reach-v0',
    entry_point='kenova_reach:KenovaReach',
    max_episode_steps=500,
)

class KenovaReach(KenovaFetchBase):
    def __init__(
        self, 
        model_path=Kenova_path, 
        render_mode="human", 
        reward_type="sparse",
        max_action_change= MAX_CTRL_CHANGE ):


        KenovaFetchBase.__init__(
        self,
        model_path, 
        frame_skip=1000,
        reward_type=reward_type,
        target_in_the_air=True,
        render_mode=render_mode,
        has_object=False,
        max_action_change=max_action_change
        )