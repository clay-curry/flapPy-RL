from math import exp, pi
from random import random
import numpy as np
from typing import List
import pygame
import time
import meta
import torch
print(torch.cuda.is_available())

# States
NUM_Y_STATES = 20                   # encodes height of player (this should be odd for keeping in center)
NUM_V_STATES = 10                   # encodes player velocity
NUM_DX_STATES = 20                   # encodes distance from pipe to player
NUM_PIPE_STATES = 8                 # encodes center position between pipes
NUM_ACTIONS = 2

# Actions
NO_FLAP = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
FLAP = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)


# Training Algorithm
USE_SARSA_LAMBDA    = True

# Tests
J = 4
LAMBDA              = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
DISCOUNT            = [0.9, 0.9, 0.9, 0.9, 0.9]
STEP_SIZE           = [0.5, 0.5, 0.5, 0.5, 0.5]


VALUES = torch.ones(
    size = (NUM_Y_STATES, NUM_V_STATES, NUM_DX_STATES, NUM_PIPE_STATES, NUM_ACTIONS)
 )
ELIGIBILITY_TRACES = torch.zeros_like(VALUES)
if meta.LOAD:
    VALUES = torch.load(f'rl/sarsa_lamb/{meta.LOAD_FILE}')

# Learns on policy
class Agent:
    def __init__(self, FPS):
        self.FPS = FPS
        self.number_frames_between_actions = int(FPS * meta.T_BETWEEN_STATES)        # used to discretize game
        self.frame_count = 0
        self.sequences = 0
        self.breakpoint_count = 0
        self.last_backup = time.time()
        self.episode = [] # (eps num, score)
        
        self.prev_SAR = None
        
        self.VALUES = VALUES
        self.ELIGIBILITY_TRACES = ELIGIBILITY_TRACES   
        print("Created agent")
                                      
    def move(self, y_pos, y_vel, x_pipe, y_pipe, score):        
        move = NO_FLAP
        self.frame_count += 1 
        self.frame_count %= self.number_frames_between_actions        
        
        if self.frame_count == 0:
            # remove past pipes
            self.score = score      
            # compute current state, reward action
            state = self.compute_state(y_pos, y_vel, x_pipe, y_pipe)
            reward = self.compute_reward(y_pos, y_pipe)
            action = self.compute_action(state, 0.05)
            self.sarsa_lambda(reward_now=reward, state_now=state, action_now=action)
            if meta.LOG:
                self.breakpoint_count = (self.breakpoint_count + 1) % meta.STATES_BETWEEN_LOG
                self.log_flappy(state=state, reward=reward, next_move=action) if self.breakpoint_count == 0 else None

            if action == 1:
                move = FLAP

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
        return 10

    def compute_action(self, state, epsilon):
        "returns the esilon-greedy action over the possible"
        # Sanity check
        if epsilon < 0 or 1 < epsilon:
            raise ValueError(f"epsilon = {epsilon} which is not in [0,1]")

        if np.random.uniform(0, 1) >= epsilon: # true (1-epsilon)% of the time
            greedy = torch.argmax(VALUES[state])
            return int(greedy)
        else:
            random = torch.randint(low=0, high=NUM_ACTIONS, size=(1,))
            return int(random)

        

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

            UPDATE = STEP_SIZE[J] * (r1 + DISCOUNT[J] * Q_now - Q_prev) * self.ELIGIBILITY_TRACES 
            self.VALUES =self.VALUES + UPDATE
            self.ELIGIBILITY_TRACES = LAMBDA[J] * STEP_SIZE[J] * self.ELIGIBILITY_TRACES
        
        self.prev_SAR = (state_now, action_now, reward_now)

    def gameover(self, score):
        now = time.time_ns() / (10 ** 9)
        print(f'GAMEOVER: score = {score}')
        self.episode.append(score)
        print(f'Number Episodes = {len(self.episode)}')
        
        if len(self.episode) >= meta.EPISODES_PER_SEQUENCE:
            self.save()
            from datetime import datetime
            # make file
            datetime = datetime.now().strftime("%m-%d-%Y %H.%M.%S") 
            if USE_SARSA_LAMBDA:
                torch.save(self.episode, f"sarsa_lamb/results/sarsa-lamb-{LAMBDA[J]}-disc-{DISCOUNT[J]}-rate-{STEP_SIZE[J]}-{datetime}")

            self.VALUES = torch.load(f'{DIR}/init')
            self.episode = []
            self.sequences += 1
            if self.sequences >= meta.SEQUENCE_PER_PARAMETER:
                exit(0)
            
        self.prev_SAR = []

    def save(self):
        if meta.SAVE == False:
            return
        from datetime import datetime
        datetime = datetime.now().strftime("%m-%d-%Y %H.%M.%S")    
        
        file_name = f"rl/sarsa_lamb/weights-{datetime}"

        # save to file
        d = self.VALUES
        torch.save(d, file_name)

    def log_flappy(self, state, reward, next_move) -> None:
        print(f'State = {state}')
        print(f'Reward = {reward}')
        print(f'Action = ' + ("Flap" if next_move == 1 else "No Flap"))
        q_no_flap = self.VALUES[state][0]
        q_flap = self.VALUES[state][1]
        print(f'V(NO_FLAP): {q_no_flap}')
        print(f'V(FLAP): {q_flap}')
        print(f'E(NO_FLAP): {self.ELIGIBILITY_TRACES[state][0]}')
        print(f'E(FLAP): {self.ELIGIBILITY_TRACES[state][1]}')
        
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
