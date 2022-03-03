from lib2to3.pgen2.token import PERCENT
import math
from tkinter import Y
from typing import List
import pygame
import time
import numpy as np

# Gameplay Attributes
SCREENWIDTH  = 288
SCREENHEIGHT = 512
BASEY        = SCREENHEIGHT * 0.79

# Bird
X_POS_AGENT  = 57
Y_MIN_AGENT  = 0                    # top of screen (0 = top)
Y_MAX_AGENT  = SCREENHEIGHT * 0.79  # where player crashes into ground
Y_MAX_VELOCITY = 10
Y_MIN_VELOCITY = -9

# Pipes
PERCENT_TOGETHER = 5
PIPEHEIGHT = 320
# PIPEGAPSIZE  = 100
# Y_MIN_LPIPE  = int(Y_MAX_AGENT * 0.2)     # smallest y-coordinate of lower pipe opening
# Y_MAX_LPIPE  = int(Y_MAX_AGENT * 0.8)     # largest y-coordinate of lower pipe opening
PIPEGAPSIZE  = BASEY * (1 - PERCENT_TOGETHER/200)
Y_MAX_LPIPE = BASEY
Y_MIN_LPIPE = 0

X_MAX_PIPE   = SCREENWIDTH - X_POS_AGENT

# (Discretized) State Attributes
NUM_Y_STATES = 10                   # encodes height of player (this should be odd for keeping in center)
NUM_V_STATES = 10                   # encodes player velocity
NUM_DX_STATES = 10                   # encodes distance from pipe to player
NUM_PIPE_STATES = 1                 # encodes center position between pipes
NUM_STATES = NUM_Y_STATES * NUM_V_STATES * NUM_DX_STATES * NUM_PIPE_STATES


# Training Hyperparameters
T_BETWEEN_STATES = 0.01
DISCOUNT = 0.3
STEP_SIZE = 0.2
N_STEPS = 1

# Checkpointing and debugging
CHECKPOINT_DT   = 5 * 60     # every 5 minutes
READ_CACHE = False
RESET_EPSILON_GREEDY = False
EPSILON_START = 0.05
STATES_BETWEEN_LOG = 1


no_flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)

# Learns on policy
class Agent:
    def __init__(self, speedup_factor=1):
        now_ts = time.time_ns() / (10 ** 9)    # used to discretize game
        self.last_state_update_ts = now_ts        # used to discretize game
        self.last_backup = now_ts
        self.episode = [] # (eps num, score)       # used for gathering data
        self.breakpoint_count = 0
        self.speedup_factor = speedup_factor
        self.prev_state: List[int] = []
        self.prev_reward: List[float] = []
        self.prev_action: List[int] = []
        self.q_state_action_epsilon = []
        self.make_vectors()
    
                                    
    def move(self, y_pos, y_vel, lpipes, upipes, score):

        print('inside move')

        now_ts = time.time_ns() / (10 ** 9)
        self.score = score
        
        d_state = now_ts - self.last_state_update_ts
        threshold = T_BETWEEN_STATES / self.speedup_factor
        action = False
        if d_state > threshold:           
            
            # always look forward, never back
            if lpipes[0]['x'] < X_POS_AGENT and len(lpipes) > 1:
                lpipes.pop()
                       
            
            state = self.compute_state(
                y_pos=y_pos, y_vel=y_vel, 
                lpipe_x=lpipes[0]['x'], lpipe_y=lpipes[0]['y'],
                upipe_y=upipes[0]['y'] + PIPEHEIGHT,
                as_index=True
                )
            
        
            reward = self.compute_reward(y_pos=y_pos,  
            lpipe_y=lpipes[0]['y'],
            upipe_y=upipes[0]['y'] + PIPEHEIGHT)
            
            self.last_state_update_ts = now_ts
            self.breakpoint_count = (self.breakpoint_count + 1) % STATES_BETWEEN_LOG
            
            if self.breakpoint_count == 0:
                print(f'Y_POS, Y_VEL, DX, C_PIPE = {state[1]}', end=" ")
                print(f'Reward = {reward}')
                print(self.q_state_action_epsilon[state[0]])
                print()
            
                
            q = self.q_state_action_epsilon[state[0]]
            # HEREEEE
            

            action = epsilon_greedy(
                q_noflap=self.q_state_action_epsilon[state[0]][0], 
                q_flap=self.q_state_action_epsilon[state[0]][1], 
                epsilon=self.q_state_action_epsilon[state[0]][2])


            self.prev_reward.append(reward)
            self.prev_action.append(action)
            self.prev_state.append(state[0])
            self.sarsa()

            



        return flap if action else no_flap

    def sarsa(self) -> bool:
        print('inside sarsa')
        if not ((np.ones(NUM_STATES) * EPSILON_START) == self.q_state_action_epsilon[:,2]).all():
                print("something happened")
        if len(self.prev_state) > N_STEPS:
            state_tn = self.prev_state.pop(0)
            action_tn = int(self.prev_action.pop(0))
            self.prev_reward.pop(0)
            Qn = self.q_state_action_epsilon[state_tn][action_tn]
            
            # Compute the discounted sum of n rewards
            G_terms = [self.prev_reward[k] * DISCOUNT**k for k in range(N_STEPS)]
            G_tn = sum(G_terms)
            
            Qn += STEP_SIZE * (G_tn - Qn)
            if not ((np.ones(NUM_STATES) * EPSILON_START) == self.q_state_action_epsilon[:,2]).all():
                print("something happened")
                for state in range(NUM_STATES):
                    if self.q_state_action_epsilon[state][2] != 0.05:
                        print(state)
                        print(self.q_state_action_epsilon[state])
            self.q_state_action_epsilon[state_tn][action_tn] = Qn
            if not ((np.ones(NUM_STATES) * EPSILON_START) == self.q_state_action_epsilon[:,2]).all():
                print("something happened")
        


    def compute_state(self, y_pos, y_vel, lpipe_x, lpipe_y, upipe_y, as_index=False):
        print('inside compute_state')
        if not ((np.ones(NUM_STATES) * EPSILON_START) == self.q_state_action_epsilon[:,2]).all():
                print("something happened")
        try:
            Y_POS = map_bin(
                x=y_pos,
                minimum=Y_MIN_AGENT-50,
                maximum=Y_MAX_AGENT,
                n_bins=NUM_Y_STATES,
                enforce_bounds=True
            )

            Y_VEL = map_bin(
                x=y_vel,
                minimum=Y_MIN_VELOCITY,
                maximum=Y_MAX_VELOCITY,
                n_bins=NUM_V_STATES
            )

            DX = map_bin(
                x=lpipe_x,
                minimum=X_POS_AGENT,
                maximum=X_MAX_PIPE,
                n_bins=NUM_DX_STATES,
                enforce_bounds=False
            )

            C_PIPE = map_bin(
                x= (upipe_y + lpipe_y)/2,
                minimum=Y_MIN_LPIPE - PIPEGAPSIZE/2 - 1,
                maximum=Y_MAX_LPIPE - PIPEGAPSIZE/2 + 1,
                n_bins=NUM_PIPE_STATES)
            

        except ValueError as e:
            print(e.with_traceback())
            raise ValueError
        index = 0
        index += C_PIPE
        index += DX * (NUM_PIPE_STATES)
        index += Y_VEL * (NUM_DX_STATES * NUM_PIPE_STATES)
        index += Y_POS * (NUM_V_STATES * NUM_DX_STATES * NUM_PIPE_STATES)
        return index, (Y_POS, Y_VEL, DX, C_PIPE)

    def compute_reward(self, y_pos, lpipe_y, upipe_y):
        print('inside compute_reward')
#return 1
        dx = (lpipe_y + upipe_y) / 2 - y_pos
        return -math.sqrt(abs(dx)) ** 6 / BASEY

    def save(self):
        print('inside save')
        from datetime import datetime
        today = datetime.now()
        d1 = today.strftime("%m-%d-%Y %H.%M.%S")
        with open(f"tdn_full/flappy_sarsa-{d1}", 'wb') as f:
            np.save(f, self.q_state_action_epsilon, allow_pickle=True)
            np.save(f, self.episode, allow_pickle=True)
            np.save(f, self.hyperparameters, allow_pickle=True)
        print(self.q_state_action_epsilon)
        print(self.episode)

    def gameover(self, score):
        print('inside gameover')
        print(f'GAMEOVER: score = {score}')
        if not ((np.ones(NUM_STATES) * EPSILON_START) == self.q_state_action_epsilon[:,2]).all():
                        raise ValueError()
        # Adjust values on remaining states
        num_states = len(self.prev_state)
        # Losing resulted in the lowest reward
        min_reward = self.q_state_action_epsilon.min() - 10
        self.prev_reward.append(min_reward)
        self.prev_reward.pop(0) # remove R_1

        discount = DISCOUNT
        R_t = 0
        G_t = 0
        for t in range(num_states-1, -1, -1):
            action = self.prev_action.pop(t)
            state = self.prev_state.pop(t)
            R_t = self.prev_reward.pop(t)  # return from Q_
            G_t = R_t + discount * G_t     # discounted return

            Qt = self.q_state_action_epsilon[state][action]
            Qt += STEP_SIZE * (G_t - Qt)
            self.q_state_action_epsilon[state][action] = Qt

        # Save last round
        self.episode.append(score)
        now = time.time_ns() / (10 ** 9)
        if now - self.last_backup > CHECKPOINT_DT:
            self.last_backup = now
            self.save()

    def make_vectors(self):
        print('inside make_vectors')
        if READ_CACHE:
            try:
                from os import listdir
                names = listdir("tdn_full")  # read most recent list
                name = names[len(names) - 1]
                with open(f"tdn_full/{name}", 'rb') as f:
                    self.q_state_action_epsilon = np.load(f, allow_pickle=True)
                    self.episode = list(np.load(f, allow_pickle=True))
                    self.hyperparameters = list(np.load(f, allow_pickle=True))

                if RESET_EPSILON_GREEDY:
                    self.q_state_action_epsilon[:, 2] = np.ones(NUM_STATES) * EPSILON_START

                print(self.q_state_action_epsilon)
                print(self.episode)
                print(self.hyperparameters)
                return
            except:
                pass

        # random
        self.hyperparameters = [
            DISCOUNT, STEP_SIZE, N_STEPS, NUM_Y_STATES, NUM_V_STATES,
            NUM_DX_STATES, NUM_PIPE_STATES, NUM_STATES, EPSILON_START
        ]
        self.q_state_action_epsilon = np.insert(
            arr=np.random.normal(loc=0, scale=1, size=(NUM_STATES,2)),
            obj=2,
            values=np.ones(NUM_STATES) * EPSILON_START,  # epsilon
            axis=1)

        for j in range(NUM_V_STATES):
            for k in range(NUM_DX_STATES):
                for g in range(NUM_PIPE_STATES):
                    for i in range(int(NUM_Y_STATES * 0.7)):
                        index = 0
                        index += g
                        index += k * (NUM_PIPE_STATES)
                        index += j * (NUM_DX_STATES * NUM_PIPE_STATES)
                        index += i * (NUM_V_STATES * NUM_DX_STATES * NUM_PIPE_STATES)
                        self.q_state_action_epsilon[index][1] = -1


def map_bin(x: float, minimum: float, maximum: float, n_bins: int,
            
            f=lambda x: x, one_indexed=False, enforce_bounds=True):
    print('inside map_bin')
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

    _hash = (x - minimum) / (maximum - minimum)
    _hash = _hash if _hash < 0.001 else _hash - 0.001
    _hash = f(_hash) * n_bins
    _hash = math.floor(_hash)
    if one_indexed:
        return _hash + 1
    else:
        return _hash


def epsilon_greedy(q_noflap, q_flap, epsilon):
    
    print('inside epsilon_greedy')
    
    if epsilon > 1:
        raise ValueError("epsilon > 1")
    if epsilon < 0:
        raise ValueError("epsilon < 0")
    # returns True for flap and False for no_flap
    if abs(q_flap - q_noflap) < 0.001:
        # if neither are greedy, pick random
        return np.random.uniform(0, 1) < .5

    # determine greedy state and return it with p = 1 - epsilon
    greedy = q_flap > q_noflap
    if np.random.uniform(0, 1) > epsilon:
        return greedy
    else:
        return not greedy




