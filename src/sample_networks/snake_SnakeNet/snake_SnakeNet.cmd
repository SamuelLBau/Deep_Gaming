python deepQ.py --env snake --proto cfn/snakenet.prototxt --game_skip 0 --n_episodes 10000000 --momentum 0.950000 --learning_rate 0.001000 --discount 0.990000 --epsilon_min 0.100000 --epsilon_max 1.000000 --epsilon_steps 2000000 --n_prev_states 100000 --checkpoint_interval 500 --target_update_interval 5000 --learning_interval 1 --minibatch_size 30 --max_neg_reward_steps 1000000000 --save_rewards 