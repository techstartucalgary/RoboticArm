mpirun -np 1 python -u train.py --env-name='Kenova_pick_and_place-v0' --render_mode='depth_array' --cuda  --n-cycles=10 2>&1 | tee reach.log  