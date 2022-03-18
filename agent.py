import numpy as np
from typing import List
import pygame
import time
import meta
import torch

# States
NUM_Y_STATES = 10                   # encodes height of player (this should be odd for keeping in center)
NUM_V_STATES = 10                   # encodes player velocity
NUM_DX_STATES = 1                   # encodes distance from pipe to player
NUM_PIPE_STATES = 1                 # encodes center position between pipes
NUM_ACTIONS = 2

# Actions
FLAP = True
NO_FLAP = False

# Values
LOAD_CACHE = False
VALUE_FLAP = torch.zeros(
    size = (NUM_Y_STATES, NUM_V_STATES, NUM_DX_STATES, NUM_PIPE_STATES)
)
VALUE_NO_FLAP = torch.ones(
    size = (NUM_Y_STATES, NUM_V_STATES, NUM_DX_STATES, NUM_PIPE_STATES)
)

VALUES = torch.ones(
    size = (NUM_Y_STATES, NUM_V_STATES, NUM_DX_STATES, NUM_PIPE_STATES, NUM_ACTIONS)
)

print(torch.ones(2,2,3))

# Hyperparameters
DISCOUNT            = 0.9
LEARNING_RATE       = 0.2
STEP_SIZE           = 0.3

# Training Algorithm
USE_N_STEP_SARSA    = True      # enable value iteration using n-step SARSA 
N_STEPS             = 5         # Q(S,A) <- Q(S,A) + LEARNING_RATE*(R1+R2+...+RN+DISCOUNT*Q(S',A')-Q(S,A))
USE_TD_LAMBDA       = False
LAMBDA              = 0.6
ELIGIBILITY_TRACE = torch.zeros(
    size = (NUM_DX_STATES, NUM_V_STATES, NUM_DX_STATES, NUM_PIPE_STATES, NUM_ACTIONS)
)


# Learns on policy
class Agent:
    def __init__(self, FPS):
        self.FPS = FPS
        self.number_frames_between_actions = int(FPS * meta.T_BETWEEN_STATES)        # used to discretize game
        self.frame_count = 0
        self.breakpoint_count = 0
        self.last_backup = time.time()
        self.episode = [] # (eps num, score)
        self.prev_SAR = (0, 0, 0)
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
                log(state=state, reward=reward, next_move=next_move) if self.breakpoint_count == 0 else None
                    

            # self.sarsa()
            self.sarsa_lambda(reward_now=reward, state_now=state, action_now=next_move)
            self.prev_SAR = (state, next_move, reward)




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
        dx = y_pipe - y_pos
        reward = -sqrt(abs(dx)) ** 3 / meta.BASEY + 10
        return reward

    def compute_action(self, state):
        q_no_flap = VALUE_NO_FLAP[state[0]][state[1]][state[2]][state[3]]
        q_flap = VALUE_FLAP[state[0]][state[1]][state[2]][state[3]]
        action = epsilon_greedy(q_no_flap, q_flap, 0)
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
        prev_state = self.prev_SAR[0]
        prev_action = self.prev_SAR[1]
        
        Q_prev = (VALUE_NO_FLAP if prev_action == NO_FLAP else VALUE_FLAP)[prev_state]
        Q_now = (VALUE_NO_FLAP if action_now == NO_FLAP else VALUE_FLAP)[state_now]

        delta = reward_now + DISCOUNT * Q_now - Q_prev
        ELIGIBILITY_TRACE[prev_state, int(prev_action)] += 1

        for y in range(NUM_Y_STATES):
            for v in range(NUM_V_STATES):
                for x in range(NUM_DX_STATES):
                    for p in range(NUM_PIPE_STATES):
                        UPDATE = LEARNING_RATE * ELIGIBILITY_TRACE[y][v][x][p][0] * delta
                        VALUE_NO_FLAP[y][v][x][p] += UPDATE
                        UPDATE = LEARNING_RATE * ELIGIBILITY_TRACE[y,v,x,p,1] * delta
                        VALUE_FLAP[y][v][x][p] += UPDATE
                        ELIGIBILITY_TRACE[y][v][x][p] *= LAMBDA * LEARNING_RATE

    def gameover(self, score):
        now = time.time_ns() / (10 ** 9)
        print(f'GAMEOVER: score = {score}')
        self.episode.append(score)
        print(f'Number Episodes = {len(self.episode)}')
        # backup values for over many training episodes
        if now - self.last_backup > meta.CHECKPOINT_DT:
            self.last_backup = now
            save(self)
       
        self.prev_SAR = []

def log(state, reward, next_move) -> None:
    print(f'State = {state}')
    print(f'Reward = {reward}')
    q_no_flap = VALUE_NO_FLAP[state]
    q_flap = VALUE_FLAP[state]
    print(f'V(NO_FLAP): {q_no_flap}')
    print(f'V(FLAP): {q_flap}')
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

def save():
    import os
    from datetime import datetime
    global VALUE_FLAP
    global VALUE_NO_FLAP

    # make file
    datetime = datetime.now().strftime("%m-%d-%Y %H.%M.%S")    
    if dir not in os.getcwd():
        os.mkdir(dir)
    file_name = f"{dir}/weights-{datetime}"

    # save to file
    d = {'NO_FLAP': VALUE_NO_FLAP, 'FLAP': VALUE_FLAP}
    torch.save(d, file_name)

if LOAD_CACHE:
    from os import listdir
    fn = listdir(DIR)
    if fn != []:    
        last = fn.pop()
        d = torch.load(last)
        VALUE_NO_FLAP = d['NO_FLAP']
        VALUE_FLAP = d['FLAP']
