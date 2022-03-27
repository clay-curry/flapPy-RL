from math import exp, pi
from operator import index
from random import random, randrange
from statistics import median
from turtle import update
import numpy as np
from typing import List
import pygame
import time
import config
import torch
import sys

# Actions
NO_FLAP = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
FLAP = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)

# States
NUM_Y_STATES = 20                   # encodes height of player (this should be odd for keeping in center)
NUM_V_STATES = 10                   # encodes player velocity
NUM_DX_STATES = 10                   # encodes distance from pipe to player
NUM_PIPE_STATES = 8                 # encodes center position between pipes
NUM_ACTIONS = 2

# Training Parameters
J = 5           # overridden by command line arguments; determines which parameter index to use
N              = [1, 3, 6, 9, 12, 15]
DISCOUNT       = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
STEP_SIZE      = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Values + Uncertainty
VALUES = torch.ones((NUM_Y_STATES, NUM_V_STATES, NUM_DX_STATES, NUM_PIPE_STATES, NUM_ACTIONS + 1)) * 3
VALUES[:,:,:,:,1] /= 1.01    # gives no-flap more value
VALUES[:,:,:,:,2] = 0.1     # certainty, initialized to 20%, reduced as points are awarded

if config.LOAD:
    VALUES = torch.load(f'data/weights/{config.LOAD_FILE}')



# Learns on policy
class Agent:
    def __init__(self, FPS):
        self.FPS = FPS
        self.t = 0        # used to discretize time
        self.score = 0
        self.sequence_count = 0
        self.Q = VALUES

        index = J
        if len(sys.argv) > 1 and sys.argv[1].isnumeric:
             index = sys.argv[1]
       
        print(index)
        self.N = N[int(index)]
        self.DISCOUNT = DISCOUNT[int(index)]
        self.STEP_SIZE = STEP_SIZE[int(index)]
        self.score_hist = []

        self.update_hist = []
        self.max_update = 1        
        self.prev_SAR = []
        print("Created agent")
                                      
    def move(self, y_pos, y_vel, x_pipe, y_pipe,score):        
        move = NO_FLAP

        self.t = (1 + self.t)       
        if self.t % int(self.FPS * config.T_BETWEEN_STATES) == 0:
            # compute current state, reward action
            
            state = self.compute_state(y_pos, y_vel, x_pipe, y_pipe)
            reward = self.compute_reward(y_pos, y_pipe)
            # action = self.compute_action(state, self.compute_epsilon(score))
            action = self.compute_action(state, self.Q[state][2])
            self.update_uncertainty(score)
            
            if config.LOG:
                self.log_flappy(state=state, reward=reward, next_move=action)
            
            # updates values
            self.update(reward_now=reward, state_now=state, action_now=action)
            
            move = FLAP if action == 1 else NO_FLAP

        return move
        
    def compute_state(self, y_pos, y_vel, x_pipe, y_pipe):
        try:
            Y_POS = map_bin(
                x=y_pos,
                minimum=config.Y_MIN_AGENT-50,
                maximum=config.Y_MAX_AGENT,
                n_bins=NUM_Y_STATES,
                enforce_bounds=True
            )

            Y_VEL = map_bin(
                x=y_vel,
                minimum=config.Y_MIN_VELOCITY,
                maximum=config.Y_MAX_VELOCITY,
                n_bins=NUM_V_STATES
            )

            DX = map_bin(
                x=x_pipe-config.X_POS_AGENT,
                minimum=0,
                maximum=config.X_MAX_PIPE,
                n_bins=NUM_DX_STATES,
                enforce_bounds=False
            )

            C_PIPE = map_bin(
                x= y_pipe-y_pos,
                minimum=config.Y_MIN_LPIPE-config.BASEY,
                maximum=config.Y_MAX_LPIPE + 30,
                n_bins=NUM_PIPE_STATES,
                enforce_bounds=True)

        except ValueError as e:
            print(e)
            raise ValueError

        return (Y_POS, Y_VEL, DX, C_PIPE)

    def compute_reward(self, y_pos, y_pipe):
        return 1
        
    def compute_action(self, state, epsilon):
        "returns the esilon-greedy action over the possible actions"
        # Sanity check
        if epsilon < 0 or 1 < epsilon:
            raise ValueError(f"epsilon = {epsilon} which is not in [0,1]")

        Q = self.Q[state]
        greedy_index = randrange(0, 2)
        if Q[0] > Q[1]:
            greedy_index = 0
        elif Q[0] < Q[1]:
            greedy_index = 1

        if np.random.uniform(0, 1) >= epsilon: # true (1-epsilon)% of the time
            return int(greedy_index)
        else:
            return 1 - greedy_index
        
    def update_uncertainty(self, score):
        if score > self.score:
            self.score = score
            for s,a,r in self.prev_SAR:
                uncertainty = self.Q[s][2] / ((1 + self.Q[s][2]) ** (1 + score ** 2))
                self.Q[s][2] = uncertainty


    def compute_epsilon(self, score):
        """determines uncertainty of agent in particular situation"""
        if score < 1: # situation: if agent struggles to find first obstacle, have high exploration until it finds it            
            recent_success = min(len(self.score_hist), 50) - self.score_hist[-50:].count(0)
            if recent_success < 50:                 # recent success indicates less exploration
                epsilon = .2 * 1 / (1 + recent_success)**(1.74027)  # exponent makes epsilon = .005 when x=20
                return epsilon
            else:
                return 0    # perfect success indicates no uncertainty

        epsilon = 0.05 * median([abs(update)/50 for update in self.update_hist[-50:]]) / self.max_update
        return epsilon
    
    def update(self, reward_now, state_now, action_now):
        self.prev_SAR.append((state_now, action_now, reward_now))
        if len(self.prev_SAR) >self.N:
            s, a, _ = self.prev_SAR.pop(0)

            Gt = [self.prev_SAR[r][2]*(self.DISCOUNT ** r) for r in range(len(self.prev_SAR))]
            value_now = self.Q[state_now][action_now]
            expected_update = sum(Gt) + value_now * self.DISCOUNT ** len(self.prev_SAR)
            
            update_size = float(self.STEP_SIZE * (expected_update - self.Q[s][a]))
            self.Q[s][a] += update_size
            self.Q[s][2] /= 1.0001
            self.update_hist.append(update_size)

    def update_gameover(self):
        while len(self.prev_SAR) > 1:
            s, a, _ = self.prev_SAR.pop(0)

            Gt = [self.prev_SAR[r][2]*(self.DISCOUNT ** r) for r in range(len(self.prev_SAR))]
            value_now = 0
            expected_update = sum(Gt) + value_now * self.DISCOUNT ** len(self.prev_SAR)
            
            update_size = float(self.STEP_SIZE * (expected_update - self.Q[s][a]))
            self.Q[s][a] += update_size
            self.Q[s][2] /= 1.01
            self.update_hist.append(update_size)
            

    def gameover(self, score):
        now = time.time_ns() / (10 ** 9)
        print(f'GAMEOVER({self.N}): score = {score}')
        self.score_hist.append(score)
        self.t=0
        print(f'Number Episodes = {len(self.score_hist)}')
        self.update_gameover()
        self.score=0

        if len(self.score_hist) >= config.EPISODES_PER_SEQUENCE:
            self.save()
            from datetime import datetime
            # make file
            datetime = datetime.now().strftime("%m-%d-%Y %H.%M.%S")     
            
            self.Q = VALUES
            self.score_hist = []
            
            self.sequence_count += 1
            if self.sequence_count >= config.SEQUENCE_PER_PARAMETER:
                exit(0)
                
    def save(self):
        if config.SAVE == False:
            return
        from datetime import datetime
        dt = datetime.now().strftime("%m-%d-%Y-%H.%M.%S")
        saved = False
        while saved == False:
            try:
                open(f"data/scores/nsarsa-{self.N}-disc-{self.DISCOUNT}-rate-{self.STEP_SIZE}-{dt}.txt", 'x') 
                with open(f"data/scores/nsarsa-{self.N}-disc-{self.DISCOUNT}-rate-{self.STEP_SIZE}-{dt}.txt", 'w') as f:
                    f.write(str(self.score_hist))

                torch.save(self.Q, f"data/weights/nsarsa-{self.N}-disc-{self.DISCOUNT}-rate-{self.STEP_SIZE}-{dt}.pt")
                saved = True
            except Exception as e:
                print(e)
                dt = datetime.now().strftime("%m-%d-%Y %H.%M.%S")

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
