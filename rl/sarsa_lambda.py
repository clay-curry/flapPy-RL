import numpy as np
from typing import List
import pygame
import time
import meta
import torch
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Actions
NO_FLAP = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
FLAP = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)

# States
NUM_Y_STATES = 20                   # encodes height of player (this should be odd for keeping in center)
NUM_V_STATES = 10                   # encodes player velocity
NUM_DX_STATES = 1                   # encodes distance from pipe to player
NUM_PIPE_STATES = 5                 # encodes center position between pipes
NUM_ACTIONS = 2

# Values
VALUES = torch.ones((NUM_Y_STATES, NUM_V_STATES, NUM_DX_STATES, NUM_PIPE_STATES, NUM_ACTIONS)) * 1.5
if meta.LOAD:
    VALUES = torch.load(f'rl/sarsa_lamb/{meta.LOAD_FILE}')

# Parameters
J = 0           # index selection for the next run
LAMBDA         = [0.1, 0.3, 0.5, 0.7, 0.9]
DISCOUNT       = [0.9, 0.9, 0.9, 0.9, 0.9]
STEP_SIZE      = [0.5, 0.5, 0.5, 0.5, 0.5]
SEQUENCE_COUNT = 0






# Learns on policy
class Agent:
    def __init__(self, FPS):
        self.FPS = FPS
        self.t = 0        # used to discretize game
        self.Q = VALUES
        self.score_hist = []
        self.update_hist = []
        
        self.prev_SAR = []
        print("Created agent")
                                      
    def move(self, y_pos, y_vel, x_pipe, y_pipe,score):        
        move = NO_FLAP

        self.t = (1 + self.t) % int(self.FPS * meta.T_BETWEEN_STATES)        
        if self.t == 0:
            # compute current state, reward action
            
            state = self.compute_state(y_pos, y_vel, x_pipe, y_pipe)
            reward = self.compute_reward(y_pos, y_pipe)
            action = self.compute_action(state, self.compute_epsilon(score))
            if meta.LOG:
                self.log_flappy(state=state, reward=reward, next_move=action)
            
            # updates values
            self.n_sarsa(reward_now=reward, state_now=state, action_now=action)
            
            move = FLAP if action == 1 else NO_FLAP

        return move
        
    def compute_state(self, y_pos, y_vel, x_pipe, y_pipe):
        try:
            Y_POS = map_bin(
                x=y_pos,
                minimum=meta.Y_MIN_AGENT-50,
                maximum=meta.Y_MAX_AGENT,
                n_bins=NUM_Y_STATES,
                enforce_bounds=True
            )

            Y_VEL = map_bin(
                x=y_vel,
                minimum=meta.Y_MIN_VELOCITY,
                maximum=meta.Y_MAX_VELOCITY,
                n_bins=NUM_V_STATES
            )

            DX = map_bin(
                x=x_pipe-meta.X_POS_AGENT,
                minimum=0,
                maximum=meta.X_MAX_PIPE,
                n_bins=NUM_DX_STATES,
                enforce_bounds=False
            )

            C_PIPE = map_bin(
                x= y_pipe,
                minimum=meta.Y_MIN_LPIPE - meta.PIPEGAPSIZE/2 - 1,
                maximum=meta.Y_MAX_LPIPE - meta.PIPEGAPSIZE/2 + 1,
                n_bins=NUM_PIPE_STATES,
                enforce_bounds=False)

        except ValueError as e:
            print(e)
            raise ValueError

        return (Y_POS, Y_VEL, DX, C_PIPE)

    def compute_reward(self, y_pos, y_pipe):
        return 1
        from math import sqrt
        sigma = 60
        dx = y_pipe - y_pos
        reward = exp(- (dx**2) / (2 * sigma**2) )
        return reward

    def compute_action(self, state, epsilon):
        "returns the esilon-greedy action over the possible"
        # Sanity check
        if epsilon < 0 or 1 < epsilon:
            raise ValueError(f"epsilon = {epsilon} which is not in [0,1]")

        Q = VALUES[state]
        greedy_indices = (Q == max(Q)).nonzero(as_tuple=True)[0]
        random_greedy_index = greedy_indices[torch.randint(len(greedy_indices), (1,))][0]
        
        if np.random.uniform(0, 1) >= epsilon: # true (1-epsilon)% of the time
            return int(random_greedy_index)
        else:
            ungreedy_indices = (Q != max(Q)).nonzero(as_tuple=True)[0]
            if len(ungreedy_indices) == 0:
                return int(random_greedy_index)
            else:
                random_ungreedy_index = ungreedy_indices[torch.randint(len(ungreedy_indices), (1,))][0]
                return int(random_ungreedy_index)

    def compute_epsilon(self, score):
        return .05 / (1 + score)

    def gameover(self, score):
        now = time.time_ns() / (10 ** 9)
        print(f'GAMEOVER: score = {score}')
        self.score_hist.append(score)
        print(f'Number Episodes = {len(self.score_hist)}')

        if len(self.score_hist) >= meta.EPISODES_PER_SEQUENCE:
            self.save()
            from datetime import datetime
            global SEQUENCE_COUNT
            global J
            # make file
            datetime = datetime.now().strftime("%m-%d-%Y %H.%M.%S")     
            torch.save(self.score_hist, f"rl/n_sarsa/results/sarsa-disc-{DISCOUNT[J]}-rate-{STEP_SIZE[J]}-{datetime}")
            self.Q = VALUES
            self.score_hist = []
            
            SEQUENCE_COUNT += 1
            if SEQUENCE_COUNT >= meta.SEQUENCE_PER_PARAMETER:
                J += 1
                if J == len(N):
                    exit(0)
                

    def save(self):
        if meta.SAVE == False:
            return
        from datetime import datetime
        datetime = datetime.now().strftime("%m-%d-%Y %H.%M.%S")    
        file_name = f"rl/n_sarsa/weights/{datetime}.pt"
        torch.save(self.Q, file_name)

    def log_flappy(self, state, reward, next_move) -> None:
        print(f'State = {state}')
        print(f'Reward = {reward}')
        print(f'Action = ' + ("Flap" if next_move == 1 else "No Flap"))
        print(f'V(NO_FLAP): {self.Q[state][0]}')
        print(f'V(FLAP): {self.Q[state][1]}')
        
        print()

def map_bin(x: float, minimum: float, maximum: float, n_bins: int,
            f=lambda x: x, one_indexed=False, enforce_bounds=True):
    # Sanity check
    if minimum > maximum:
        raise ValueError("minimum was not less than maximum")
    elif n_bins <= 0:
        raise ValueError("number of bins in positive")
    elif x < minimum:
        if enforce_bounds:
            raise ValueError("x was less than minimim")
        else:
            x = minimum
    elif x > maximum:
        if enforce_bounds:
            raise ValueError("x was greater than maximum")
        else:
            x = maximum

    # map to bin
    from math import floor
    _hash = (x - minimum) / (maximum - minimum)
    _hash = _hash if _hash < 0.0000001 else _hash - 0.0000001
    _hash = f(_hash) * n_bins
    _hash = floor(_hash)
    if one_indexed:
        return _hash + 1
    else:
        return _hash


