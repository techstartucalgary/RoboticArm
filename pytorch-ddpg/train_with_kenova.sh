python pytorch-ddpg/main.py --env Kenova_pick_and_place-v0 --max_episode_length 200 --ou_sigma 0.5 --debug --validate_episodes=1
python pytorch-ddpg/main.py --env Kenova_reach-v0 --max_episode_length 200 --ou_sigma 0.5 --debug --validate_episodes=1 --resume=Kenova_reach-v0-run1


python pytorch-ddpg/main.py --mode test --env Kenova_reach-v0 --validate_episodes 100 --max_episode_length 500 --ou_sigma 0.5  --validate_episodes=20 --resume=pytorch-ddpg/output/Kenova_reach-v0-run1 --debug
