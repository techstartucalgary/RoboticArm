python pytorch-ddpg/main.py --env Kenova_pick_and_place-v0 --max_episode_length 200 --ou_sigma 0.5 --debug --validate_episodes=1
python pytorch-ddpg/main.py --env Kenova_reach-v0 --max_episode_length 200 --ou_sigma 0.5 --debug --validate_episodes=1 --resume=Kenova_reach-v0-run3

python pytorch-ddpg/main.py --env Kenova_slide-v0 --max_episode_length 300 --ou_sigma 0.5 --debug --validate_episodes=1 --hidden1=100 --hidden2=100



### Test
python pytorch-ddpg/main.py --mode test --env Kenova_reach-v0 --validate_episodes 100 --max_episode_length 200 --ou_sigma 0.5  --validate_episodes=20 --resume=pytorch-ddpg/output/Kenova_reach-v0-run1 --debug
# testing using run 16
python pytorch-ddpg/main.py --mode test --env Kenova_reach-v0 --validate_episodes 100 --max_episode_length 200 --ou_sigma 0.5  --validate_episodes=20 --resume=pytorch-ddpg/output/Kenova_reach-v0-run16 --debug --hidden1=100 --hidden2=100

# stage 1
## fixed reach
python pytorch-ddpg/main.py --mode train --env Reach_stage1 --validate_episodes 20 --max_episode_length 200 --ou_sigma 0.5   --hidden1=100 --hidden2=100 --debug

## fixed reach test
python pytorch-ddpg/main.py --mode test --env Reach_stage1 --validate_episodes 20 --max_episode_length 200 --ou_sigma 0.5   --hidden1=100 --hidden2=100 --debug --resume=pytorch-ddpg/output/stage1_1/succeed_model

# stage 2
## train
python pytorch-ddpg/main.py --mode train --env Reach_stage2 --validate_episodes 20 --max_episode_length 200 --ou_sigma 0.5   --hidden1=100 --hidden2=100 --debug --init_arm_pos=pytorch-ddpg/output/stage1_1/succeed_pos
## test
python pytorch-ddpg/main.py --mode test --env Reach_stage2 --validate_episodes 20 --max_episode_length 200 --ou_sigma 0.5   --hidden1=100 --hidden2=100 --debug --init_arm_pos=pytorch-ddpg/output/stage1_1/succeed_pos --resume=pytorch-ddpg/output/stage2_1/succeed_model