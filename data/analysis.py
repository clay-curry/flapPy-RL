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
    directory_name = 'data/scores/'
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
        ax.set_title(f"{n_step}-step SARSA")
        plt.scatter(x=x, y=episode_score_list[0], s=0.1, c='#ff0000', marker='*', label='Run 1')
        plt.scatter(x=x, y=episode_score_list[1], s=0.1, c='#0000ff', marker='*', label='Run 2')
        plt.legend()
        ax.set_ylabel("Score (stopped at 1000)")
        ax.set_xlabel("Episodes")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 1000)
    
    plt.show()



if __name__ == "__main__":
    n_rewards = load_nsarsa_reward_lists()
    dictionary_to_vertical_scatter_plot(n_rewards)



