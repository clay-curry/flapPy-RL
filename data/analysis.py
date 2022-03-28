from cgi import test
import json
from random import random, randrange
from statistics import median
from tkinter import font
from turtle import color
from typing import Dict, List
import numpy as np
import re
import os
import torch
import matplotlib.pyplot as plt
episodes = []
colors=['b', 'g', 'r', 'c', 'y', 'm']

def load_scores():
    """returns a dict of the form: {n: [reward_list1, reward_list_2, ...]}"""
    directory_name = 'data/visualizations/'
    file_names = os.listdir(directory_name)
    scores = []
    for file in file_names:
        with open(directory_name + file, 'r') as f:
            scores.append(json.load(f))
    
    return scores


def score_scatter(data: List[List[int]]):
    fig, ax = plt.subplots()
    label_font = {'family': 'serif',
        'color':  'Black',
        'weight': 'normal',
        'size': 14,
        }
    title_font = label_font.copy()
    title_font['size'] = 16
    ax.set_title(r"$\pi$ Iteration with Q-Learning ($\alpha$ = 0.2)", fontdict=title_font,pad=10)
    ax.set_ylabel("Score (stopped at 300)", fontdict=label_font,labelpad=4)
    ax.set_xlabel("Episodes", fontdict=label_font)
    ax.set_ylim(0, 300)
    ax.set_yticks([0, 70, 140, 210, 280])
    ax.set_xlim(0, 40000)
    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    ax.xaxis.tick_top()
    
# max line
    k = 100
    x_max = [j for j in range(0, len(data[0]), k)]
    y_max = [0]
    for x in x_max[1:]:        
        samples = []
        for run in range(len(data)):
            samples.extend(data[run][x:x+k])
        
        y_max.append(max(max(y_max),max(samples)))

    plt.plot(x_max,y_max,c='#000000', label=f'Max')


    # mean line
    from statistics import mean
    k = 100
    x_mean = [j for j in range(0, len(data[0]), k)]
    y_mean = []
    for x in x_mean:
        samples = []
        for run in range(len(data)):
            samples.extend(data[run][x:x+k])
        y_mean.append(mean(samples))

    plt.plot(x_mean,y_mean,c='#ff0000', label=f'Mean')
    
    # median line
    from statistics import median
    k = 1000
    x_median = [j for j in range(0, len(data[0]), k)]
    y_median = []
    for x in x_median:
        samples = []
        for run in range(len(data)):
            samples.extend(data[run][x:x+k])
        y_median.append(median(samples))

    plt.plot(x_median,y_median,c='#0000ff', label=f'Median')

    # create legend
    [plt.scatter([0],[0], marker='*', s=30, c=colors[j], label=f'Agent {j+1}') for j in range(len(data))]
    plt.legend()

    high_scores = [max(run) for run in data]
    high_score_first_occurance = [run.index(hi) for hi, run in zip(high_scores, data)]
    
    x = max(high_score_first_occurance)
    y = high_scores[high_score_first_occurance.index(x)]

    ax.annotate(
        '$\pi_*$ obtained\n$\\forall$ agents',
        xy=(x, y), xycoords='data',
        xytext=(-150, -65), textcoords='offset points',
        arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc,angleA=0,armA=01,angleB=-90,armB=30,rad=7"))

    # plot random points
    import random
    x_points = [random.sample(range(len(data[i])), len(data[i])) for i in range(len(data))]
    while max([len(p) for p in x_points]) != 0:
        run = random.choice([i for i in range(len(x_points)) if len(x_points[i]) > 0])
        x = x_points[run][-50:]
        y = [data[run][i] for i in x]
        x_points[run] = x_points[run][:-50]
        plt.scatter(x,y,marker='*', s=.1, c=colors[run])

    

    plt.show()



if __name__ == "__main__":
    scores = load_scores()
    score_scatter(scores)



