import gym
import os, sys
base_file = os.path.join(os.path.dirname(__file__),'..')
sys.path.append(base_file)
from kenova_fetch_base import KenovaFetchBase

Kenova_path = "Kenova_2f85_pick_place.xml"

gym.envs.register(
    id='Kenova_push-v0',
    entry_point='kenova_push:KenovaPush',
    max_episode_steps=500,
)

class KenovaPush(KenovaFetchBase):
    def __init__(
        self, 
        model_path=Kenova_path, 
        render_mode="human", 
        reward_type="sparse"):

        KenovaFetchBase.__init__(
            self,
        model_path, 
        frame_skip=1000,
        reward_type=reward_type,
        target_in_the_air=False,
        render_mode=render_mode,
        has_object=True,
        )