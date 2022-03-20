from cgi import test
import json
from typing import Dict, List
import numpy as np
import re
import os
import torch
import matplotlib.pyplot as plt
episodes = []


def load_nsarsa_reward_lists():
    """returns a dict of the form: {n: [reward_list1, reward_list_2, ...]}"""
    directory_name = 'rl/n_sarsa/scores/'
    file_names = os.listdir(directory_name)
    meta = []
    reward_dict = {}
    for file in file_names:
        with open(directory_name + file, 'r') as f:
            n_disc_rate = [re.findall(r'[\d\.]+', file)[i] for i in range(3)]
            rewards = json.load(f)
            reward_dict.setdefault(n_disc_rate[0], []).append(rewards)
    
    return reward_dict


def dictionary_to_vertical_scatter_plot(data: Dict[int, List[List[int]]]):
    num_plots = len(data.keys())
    
    for index, (n_step, episode_score_list) in zip(range(num_plots), data.items()):
        
        x = [i for i in range(len(episode_score_list[0]))]
        fig, ax = plt.subplots()
        ax.set_title("title")
        plt.scatter(x=x, y=episode_score_list[0], s=0.1, marker='o', label='Run 1')
        plt.scatter(x=x, y=episode_score_list[1], s=0.1, marker='o', label='Run 2')
        plt.scatter(x=x, y=episode_score_list[2], s=0.1, marker='o', label='Run 3')
        plt.legend()
    
    plt.show()



if __name__ == "__main__":
    n_rewards = load_nsarsa_reward_lists()
    dictionary_to_vertical_scatter_plot(n_rewards)




"""file_names = os.listdir("visualize")
file_names.pop(file_names.index('visualize.py'))
print(file_names)
for file in file_names:
    with open(f"visualize/{file}", 'rb') as f:
        q_state_action_epsilon = np.load(f, allow_pickle=True).tolist()
        episode = np.load(f, allow_pickle=True).tolist()
        episodes.append(episode)
        hyperparameters = np.load(f, allow_pickle=True).tolist()

x = [x+1 for x in range(31)]
print(len(episodes[0]))
plt.scatter(x, episodes[0], color='r', label='Trial 1 (1000 episodes)')
plt.scatter(x, episodes[1], color='b', label='Trial 2 (1000 episodes)')
plt.scatter(x, episodes[2], color='g', label='Trial 3 (1000 episodes)')
plt.scatter(x, episodes[3], color='y', label='Trial 4 (1000 episodes)')
plt.scatter(x, episodes[4], color='m', label='Trial 5 (1000 episodes)')
plt.ylabel("Score (stopped at 1000)")
plt.xlabel("Episodes")
plt.title(r"$n$-step Sarsa ($n$=5, $\gamma$=0.2, $a$=0.3)")
plt.legend()
plt.show()"""