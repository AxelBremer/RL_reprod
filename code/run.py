import train
import argparse
from types import SimpleNamespace

num_of_runs = 5
environments = ['M', 'A', 'G']
buffer_size = [1000, 10000, 100000]


def create_config(run_config, env, bs, run_num):
    single_run_config = SimpleNamespace(**vars(run_config))
    setattr(single_run_config, 'environment', env)
    setattr(single_run_config, 'replay_capacity', bs)
    setattr(single_run_config, 'name', f'{run_config.name_prefix}_{env}_{bs}_{run_num}')
    return single_run_config

def main(run_config):
    for env in environments:
        for bs in buffer_size:
            for i in range(num_of_runs):
                train_config = create_config(run_config, env, bs, i + 1)
                print(f'Starting trainin on: {train_config.name}')
                train.main(train_config)
 

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--name_prefix', type=str, required=True, help='prefix of name run')
    parser.add_argument('--replay_type', type=str, required=True, help='Replay type: [S]tandard, [H]indsight, [P]rioritized')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden unit')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for Adam')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to train on')
    parser.add_argument('--render', type=bool, default=False, help='Boolean to render environment or not')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')

    run_config = parser.parse_args()

    main(run_config)
