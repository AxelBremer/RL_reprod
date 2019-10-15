import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_context('poster')

   # Parse training configuration
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='runs', help='folder name')
config = parser.parse_args()

scores = {'env':[], 'replay_type':[], 'buffer_size':[], 'parameters':[], 'reward':[], 'episode':[]}

for env in os.listdir(config.folder):
    for er in os.listdir(f'{config.folder}/{env}'):
        for buff in os.listdir(f'{config.folder}/{env}/{er}'):
            for params in os.listdir(f'{config.folder}/{env}/{er}/{buff}'):
                for run in os.listdir(f'{config.folder}/{env}/{er}/{buff}/{params}'):
                    with open(f'{config.folder}/{env}/{er}/{buff}/{params}/{run}/history.json', 'r') as f:
                        d = json.load(f)
                    rewards = pd.Series(d['rewards']).rolling(10).mean().values
                    eps = list(range(len(rewards)))
                    scores['env'].extend([env for x in rewards])
                    scores['replay_type'].extend([er for x in rewards])
                    scores['buffer_size'].extend([buff for x in rewards])
                    scores['parameters'].extend([params for x in rewards])
                    scores['reward'].extend(rewards)
                    scores['episode'].extend(eps)

scores = pd.DataFrame(scores)

print(scores.head())


for env in scores['env'].unique():
    part = scores[scores['env'] == env]
    fig, axes = plt.subplots(1, 3, figsize=(15,10))
    rts = part['replay_type'].unique()
    for i,rt in enumerate(rts):
        print(rt)
        p = part[part['replay_type']==rt]
        sns.lineplot(ax=axes[i], data=p, x='episode', y='reward', hue='buffer_size')
        axes[i].set_ylim(0,450)
        axes[i].title.set_text(rt)
    fig.suptitle(env, fontsize=16)
    plt.show()