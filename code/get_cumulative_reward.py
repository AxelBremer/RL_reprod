import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')

   # Parse training configuration
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='runs', help='folder name')
config = parser.parse_args()

scores = {'env':[], 'replay_type':[], 'buffer_size':[], 'parameters':[], 'cumulative_reward':[]}

for env in os.listdir(config.folder):
    for er in os.listdir(f'{config.folder}/{env}'):
        for buff in os.listdir(f'{config.folder}/{env}/{er}'):
            for params in os.listdir(f'{config.folder}/{env}/{er}/{buff}'):
                for run in os.listdir(f'{config.folder}/{env}/{er}/{buff}/{params}'):
                    try:
                        with open(f'{config.folder}/{env}/{er}/{buff}/{params}/{run}/history.json', 'r') as f:
                            d = json.load(f)
                        scores['env'].append(env)
                        scores['replay_type'].append(er)
                        scores['buffer_size'].append(buff)
                        scores['parameters'].append(params)
                        scores['cumulative_reward'].append(np.array(d['rewards']).sum())
                    except:
                        pass

scores = pd.DataFrame(scores)
for env in scores['env'].unique():
    part = scores[scores['env'] == env]
    fig, ax = plt.subplots(figsize=(15,10))
    sns.barplot(ax=ax, data=part, x='parameters', y='cumulative_reward', hue='buffer_size')
    ax.set_title(env)
    plt.show()