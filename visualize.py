import numpy as np
import os
import torch
import matplotlib.pyplot as plt
episodes = []



file_names = os.listdir("visualize")
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
plt.show()