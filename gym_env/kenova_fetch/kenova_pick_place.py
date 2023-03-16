import gym
import sys,os
base_file = os.path.join(os.path.dirname(__file__),'..')
sys.path.append(base_file)
from kenova_fetch_base import KenovaFetchBase

Kenova_path = "Kenova_2f85_pick_place.xml"

gym.envs.register(

    id='Kenova_pick_and_place-v0',
    entry_point='kenova_pick_place:KenovaPickAndPlace',
    max_episode_steps=500,
)

class KenovaPickAndPlace(KenovaFetchBase):
    def __init__(
        self, 
        model_path=Kenova_path, 
        render_mode="human", 
        reward_type='sparse'):


        print(reward_type)
        KenovaFetchBase.__init__(
            self,
            model_path, 
            frame_skip=1000,
            reward_type=reward_type,
            target_in_the_air=True,
            render_mode=render_mode,
            has_object=True,
        )