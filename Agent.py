import numpy as np
from typing import List
import pygame
import time

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
T_BETWEEN_STATES = 0.2
DISCOUNT = 0.2
STEP_SIZE = 0.3
N_STEPS = 5

# Checkpointing and debugging
CHECKPOINT_DT   = 5 * 60     # every 5 minutes
READ_CACHE = False
RESET_EPSILON_GREEDY = False
EPSILON_START = 0.05
STATES_BETWEEN_LOG = 1

EPSILON = (np.ones(NUM_STATES) * EPSILON_START).tolist()

FLAP = True
NO_FLAP = False
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
        if READ_CACHE:
            self.read_cache()
        if self.q_state_action_epsilon == []:
            self.make_vectors()
        print("Created agent")
    
                                    
    def move(self, y_pos, y_vel, lpipes, upipes, score):        
        next_move = NO_FLAP
        # remove past pipes
        if lpipes[0]['x'] < X_POS_AGENT and len(lpipes) > 1:
                lpipes.pop()

        now_ts = time.time_ns() / (10 ** 9)        
        state_dt = now_ts - self.last_state_update_ts
        if state_dt > T_BETWEEN_STATES / self.speedup_factor:           
            self.last_state_update_ts = now_ts
            self.score = score       
            state = self.compute_state(
                y_pos=y_pos, y_vel=y_vel, 
                lpipe_x=lpipes[0]['x'], lpipe_y=lpipes[0]['y'],
                upipe_y=upipes[0]['y'] + PIPEHEIGHT,
                as_index=True
                )

            reward = self.compute_reward(y_pos=y_pos,  
                lpipe_y=lpipes[0]['y'],
                upipe_y=upipes[0]['y'] + PIPEHEIGHT
                )   
                             
            next_move = epsilon_greedy(
                q_noflap=self.q_state_action_epsilon[state[0]][0], 
                q_flap=self.q_state_action_epsilon[state[0]][1], 
                epsilon=self.q_state_action_epsilon[state[0]][2])


            self.prev_reward.append(reward)
            self.prev_action.append(next_move)
            self.prev_state.append(state[0])
            
            self.breakpoint_count = (self.breakpoint_count + 1) % STATES_BETWEEN_LOG
            if self.breakpoint_count == 0:
                print(f'Y_POS, Y_VEL, DX, C_PIPE = {state[1]} ({state[0]})', end=" ")
                print(f'Reward = {reward}')
                print(self.q_state_action_epsilon[state[0]])
                print(next_move)
                print()

            self.sarsa()

        no_flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
        flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
        return flap if next_move else no_flap

    def sarsa(self) -> bool:
            if [e[2] for e in self.q_state_action_epsilon] != EPSILON:
                raise ValueError
            if len(self.prev_state) > N_STEPS:
                state_tn = self.prev_state.pop(0)
                action_tn = int(self.prev_action.pop(0))
                self.prev_reward.pop(0)
                Qn = self.q_state_action_epsilon[state_tn][action_tn]
                
                # Compute the discounted sum of n rewards
                G_terms = [self.prev_reward[k] * DISCOUNT**k for k in range(N_STEPS)]
                G_tn = sum(G_terms)
                
                Qn += STEP_SIZE * (G_tn + - Qn)
                self.q_state_action_epsilon[state_tn][action_tn] = Qn
            if [e[2] for e in self.q_state_action_epsilon] != EPSILON:
                raise ValueError


    def compute_state(self, y_pos, y_vel, lpipe_x, lpipe_y, upipe_y, as_index=False):
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
        #return 1
        from math import sqrt
        dx = (lpipe_y + upipe_y) / 2 - y_pos
        return -sqrt(abs(dx)) ** 3 / BASEY + 10

    def save(self):
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
        print(f'GAMEOVER: score = {score}')
        if [e[2] for e in self.q_state_action_epsilon] != EPSILON:
                raise ValueError
        # Adjust values on remaining states
        num_states = len(self.prev_state)
        min_reward = 0
        print(f'min reward = {min_reward}')
        self.prev_reward.append(min_reward)
        self.prev_reward.pop(0) # remove R_1

        R_t = 0
        G_t = 0
        for t in range(num_states-1, -1, -1):
            action = self.prev_action.pop(t)
            state = self.prev_state.pop(t)
            R_t = self.prev_reward.pop(t)  # return from Q_
            G_t = R_t + DISCOUNT * G_t     # discounted return
            Qt = self.q_state_action_epsilon[state][action]
            Qt += STEP_SIZE * (G_t - Qt)
            self.q_state_action_epsilon[state][action] = Qt
            
        # Save last round
        self.episode.append(score)
        now = time.time_ns() / (10 ** 9)
        if now - self.last_backup > CHECKPOINT_DT:
            self.last_backup = now
            self.save()

        if [e[2] for e in self.q_state_action_epsilon] != EPSILON:
                raise ValueError

    def make_vectors(self):
        # random
        self.hyperparameters = [
            DISCOUNT, STEP_SIZE, N_STEPS, NUM_Y_STATES, NUM_V_STATES,
            NUM_DX_STATES, NUM_PIPE_STATES, NUM_STATES, EPSILON_START
        ]
        self.q_state_action_epsilon = np.insert(
            arr=np.random.normal(loc=10, scale=1, size=(NUM_STATES,2)),
            obj=2,
            values=np.ones(NUM_STATES) * EPSILON_START,  # epsilon
            axis=1).tolist()

        """for j in range(NUM_V_STATES):
            for k in range(NUM_DX_STATES):
                for g in range(NUM_PIPE_STATES):
                    for i in range(int(NUM_Y_STATES * 0.7)):
                        index = 0
                        index += g
                        index += k * (NUM_PIPE_STATES)
                        index += j * (NUM_DX_STATES * NUM_PIPE_STATES)
                        index += i * (NUM_V_STATES * NUM_DX_STATES * NUM_PIPE_STATES)
                        self.q_state_action_epsilon[index][1] = -10"""

    def read_cache(self):
        try:
            from os import listdir
            names = listdir("tdn_full")  # read most recent list
            name = names[len(names) - 1]
            with open(f"tdn_full/{name}", 'rb') as f:
                self.q_state_action_epsilon = np.load(f, allow_pickle=True).tolist()
                self.episode = np.load(f, allow_pickle=True).tolist()
                self.hyperparameters = np.load(f, allow_pickle=True).tolist()

            if RESET_EPSILON_GREEDY:
                self.q_state_action_epsilon[:, 2] = np.ones(NUM_STATES) * EPSILON_START

            print(self.q_state_action_epsilon)
            print(self.episode)
            print(self.hyperparameters)
            return
        except:
            pass

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




