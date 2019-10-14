import train
import argparse
from types import SimpleNamespace

num_of_runs = 5
buffer_size = [3000, 10000, 30000]
replay_types = ['S', 'H', 'P']
environments = ['G']
# dfs = [0.99, 0.8, 0.75, 0.7]
# dfs = [0.75, 0.7]
# lrs = [0.0005]
# lrs = [0.0001]

def create_config(run_config, bs, rt):
    single_run_config = SimpleNamespace(**vars(run_config))
    setattr(single_run_config, 'replay_capacity', bs)
    setattr(single_run_config, 'replay_type', rt)
    setattr(single_run_config, 'name', f'{run_config.name_prefix}_{rt}_{bs}')
    return single_run_config

def create_config_gs(run_config, lr, df):
    single_run_config = SimpleNamespace(**vars(run_config))
    setattr(single_run_config, 'learning_rate', lr)
    setattr(single_run_config, 'discount_factor', df)
    setattr(single_run_config, 'name', f'lr{str(lr).split(".")[1]}_df{str(df).split(".")[1]}')
    return single_run_config

def main(run_config):
    for bs in buffer_size:
        for rt in replay_types:
            for i in range(num_of_runs):
                train_config = create_config(run_config, bs, rt)
                print(f'Starting trainin on: {train_config.name}')
                train.main(train_config)

def main2(run_config):
    for lr in lrs:
        for df in dfs:
            for i in range(num_of_runs):
                train_config = create_config_gs(run_config, lr, df)
                print(f'Starting training on: {train_config.name}')
                train.main(train_config) 

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--name_prefix', type=str, required=True, help='prefix of name run')
    # parser.add_argument('--replay_type', type=str, default='S', help='Replay type: [S]tandard, [H]indsight, [P]rioritized')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden unit')
    parser.add_argument('--environment', type=str, required=True, help='What environment to use: [M]ountainCar, [A]crobot, [C]artpole, [G]ridworld')
    parser.add_argument('--replay_capacity', type=int, default=10000, help='Number of moves to save in replay memory')

    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for Adam')
    parser.add_argument('--num_episodes', type=int, default=300, help='Number of episodes to train on')
    parser.add_argument('--render', type=bool, default=False, help='Boolean to render environment or not')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--replay_k', type=int, default=4, help='In the case of HER, the ratio of HER replays vs normal replays')
    parser.add_argument('--num_until', type=int, default=1000, help='Number of steps for epsilon to be 0.05')

    run_config = parser.parse_args()

    main(run_config)
