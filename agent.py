from math import exp, pi
import numpy as np
from typing import List
import pygame
import time
import meta
import torch
print(torch.cuda.is_available())

# States
NUM_Y_STATES = 10                   # encodes height of player (this should be odd for keeping in center)
NUM_V_STATES = 10                   # encodes player velocity
NUM_DX_STATES = 10                   # encodes distance from pipe to player
NUM_PIPE_STATES = 8                 # encodes center position between pipes
NUM_ACTIONS = 2

# Actions
FLAP = True
NO_FLAP = False


# Hyperparameters
DISCOUNT            = 0.9
LEARNING_RATE       = 0.2
STEP_SIZE           = 0.3

# Training Algorithm
USE_N_STEP_SARSA    = True      # enable value iteration using n-step SARSA 
N_STEPS             = 5         # Q(S,A) <- Q(S,A) + LEARNING_RATE*(R1+R2+...+RN+DISCOUNT*Q(S',A')-Q(S,A))
USE_TD_LAMBDA       = False
LAMBDA              = 0.8

VALUES = torch.ones(
    size = (NUM_Y_STATES, NUM_V_STATES, NUM_DX_STATES, NUM_PIPE_STATES, NUM_ACTIONS)
 )
ELIGIBILITY_TRACES = torch.zeros_like(VALUES)


# Learns on policy
class Agent:
    def __init__(self, FPS):
        self.FPS = FPS
        self.number_frames_between_actions = int(FPS * meta.T_BETWEEN_STATES)        # used to discretize game
        self.frame_count = 0
        self.breakpoint_count = 0
        self.last_backup = time.time()
        self.episode = [] # (eps num, score)
        self.prev_SAR = None
        
        self.VALUES = VALUES
        self.ELIGIBILITY_TRACES = ELIGIBILITY_TRACES   
        print("Created agent")
                                      
    def move(self, y_pos, y_vel, x_pipe, y_pipe, score):        
        next_move = NO_FLAP
        self.frame_count += 1 
        self.frame_count %= self.number_frames_between_actions        
        
        if self.frame_count == 0:
            # remove past pipes
            self.score = score       
            state = self.compute_state(y_pos, y_vel, x_pipe, y_pipe)
            reward = self.compute_reward(y_pos, y_pipe)
            next_move = self.compute_action(state)
            
            if meta.LOG:
                self.breakpoint_count = (self.breakpoint_count + 1) % meta.STATES_BETWEEN_LOG
                self.log(state=state, reward=reward, next_move=next_move) if self.breakpoint_count == 0 else None
                    
            # self.sarsa()
            self.sarsa_lambda(reward_now=reward, state_now=state, action_now=next_move)

        no_flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
        flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
        return flap if next_move else no_flap
        
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
                x=x_pipe,
                minimum=meta.X_POS_AGENT,
                maximum=meta.X_MAX_PIPE,
                n_bins=NUM_DX_STATES,
                enforce_bounds=False
            )

            C_PIPE = map_bin(
                x= y_pipe,
                minimum=meta.Y_MIN_LPIPE - meta.PIPEGAPSIZE/2 - 1,
                maximum=meta.Y_MAX_LPIPE - meta.PIPEGAPSIZE/2 + 1,
                n_bins=NUM_PIPE_STATES)

        except ValueError as e:
            print(e.with_traceback())
            raise ValueError

        return (Y_POS, Y_VEL, DX, C_PIPE)

    def compute_reward(self, y_pos, y_pipe):
        #return 1
        from math import sqrt
        sigma = 60
        dx = y_pipe - y_pos
        reward = exp(- (dx**2) / (2 * sigma**2) )
        return reward

    def compute_action(self, state):
        q_no_flap = self.VALUES[state][0]
        q_flap = self.VALUES[state][1]
        action = epsilon_greedy(q_no_flap, q_flap, 0.05)
        return action

    def sarsa(self):
        if len(self.prev_state) > N_STEPS:
            state_tn = self.prev_state.pop(0)
            action_tn = int(self.prev_action.pop(0))
            self.prev_reward.pop(0)
            Q1 = self.q_state_action_epsilon[state_tn][action_tn]
            
            # Compute the discounted sum of n rewards
            G_terms = [self.prev_reward[k] * DISCOUNT**k for k in range(N_STEPS)]
            G_tn = sum(G_terms)
            QN = max(self.q_state_action_epsilon[self.prev_state[N_STEPS-1]][0:2])

            Q1 = Q1 + STEP_SIZE * (G_tn + (DISCOUNT**N_STEPS)*QN - Q1)
            self.q_state_action_epsilon[state_tn][action_tn] = Q1

    def sarsa_lambda(self, reward_now, state_now, action_now):
        if self.prev_SAR is not None and len(self.prev_SAR) > 0:
            s0 = self.prev_SAR[0]
            a0 = self.prev_SAR[1]
            r1 = reward_now
            s1 = state_now
            a1 = action_now
            
            Q_prev =self.VALUES[s0][int(a0)]
            Q_now =self.VALUES[s1][int(a1)]
            
            self.ELIGIBILITY_TRACES[s0][int(a0)] += 1

            UPDATE = LEARNING_RATE * (r1 + DISCOUNT * Q_now - Q_prev) * self.ELIGIBILITY_TRACES 
            self.VALUES =self.VALUES + UPDATE
            self.ELIGIBILITY_TRACES = LAMBDA * LEARNING_RATE * self.ELIGIBILITY_TRACES
        
        self.prev_SAR = (state_now, action_now, reward_now)

    def gameover(self, score):
        now = time.time_ns() / (10 ** 9)
        print(f'GAMEOVER: score = {score}')
        self.episode.append(score)
        print(f'Number Episodes = {len(self.episode)}')
        # backupself.VALUES for over many training episodes
        if now - self.last_backup > meta.CHECKPOINT_DT:
            self.last_backup = now
            self.save()
       
        self.prev_SAR = []

    def save(self):
        import os
        from datetime import datetime
        # make file
        datetime = datetime.now().strftime("%m-%d-%Y %H.%M.%S")    
        if DIR not in os.listdir():
            os.mkdir(DIR)
        file_name = f"{DIR}/weights-{datetime}"

        # save to file
        d = self.VALUES
        torch.save(d, file_name)

    def log(self, state, reward, next_move) -> None:
        print(f'State = {state}')
        print(f'Reward = {reward}')
        q_no_flap = self.VALUES[state][0]
        q_flap = self.VALUES[state][1]
        print(f'V(NO_FLAP): {q_no_flap}')
        print(f'V(FLAP): {q_flap}')
        print(f'E(NO_FLAP): {self.ELIGIBILITY_TRACES[state][0]}')
        print(f'E(FLAP): {self.ELIGIBILITY_TRACES[state][1]}')
        print("Flap" if next_move else "No Flap")
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

def epsilon_greedy(q_noflap, q_flap, epsilon):    
    # Sanity check
    if epsilon < 0 or 1 < epsilon:
        raise ValueError(f"epsilon = {epsilon} which is not in [0,1]")

    # Determine greedy state
    greedy = NO_FLAP
    if q_noflap < q_flap:
        greedy = FLAP

    # Return greedy state with p = 1 - epsilon
    if np.random.uniform(0, 1) < epsilon:
        return not greedy
    else:
        return greedy

# Determine the save directory
DIR = ""
if USE_N_STEP_SARSA:
    DIR = 'sarsa'
elif USE_TD_LAMBDA:
    DIR = 'td-lambda'

if meta.LOAD:
    from os import listdir
    fn = listdir(DIR)
    if fn != []:    
        last = fn.pop()
        VALUES = torch.load(f'{DIR}/{last}')
        
