import math
import random
import time

import numpy.random
import pygame
import numpy as np
from typing import List, Tuple

READ_CACHE = True
RESET_EPSILON_START = -1 # -1 FALSE

t_epsilon = 0.001  # sec
t_between_states = 0.05  # sec
states_between_breakpoints = 3

# hyperparameters
NUM_Y_STATES = 11  # encodes height of player (this should odd for maintaining center)
NUM_V_STATES = 10  # encodes player velocity
NUM_DX_STATES = 6  # encodes distance from pipe to player
NUM_PIPE_STATES = 1  # encodes center position between pipes
NUM_STATES = NUM_Y_STATES * NUM_V_STATES * NUM_DX_STATES * NUM_PIPE_STATES


no_flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)


class TDNSarsaFull:
    def __init__(self, N=1, speedup_factor=1, xpos=57, ypos_min=0, ypos_max=100,
                 yvel_min=-10, yvel_max=10, dx_max=50, pipec_min=0, pipec_max=100,
                 step_size=0.4, discount=0.3):
        self.now_ts = time.time_ns() / (10 ** 9)
        self.N = N
        self.breakpoint_count = 0
        self.speedup_factor = speedup_factor
        self.last_state_update_ts = self.now_ts
        self.last_flap = self.now_ts
        self.episode = []
        self.pos_x = xpos
        self.ypos_min = ypos_min
        self.ypos_max = ypos_max
        self.yvel_min = yvel_min
        self.yvel_max = yvel_max
        self.dx_max = dx_max
        self.pipec_min = pipec_min
        self.pipec_max = pipec_max
        self.pipec_max = pipec_max
        self.pipe_height = 320
        self.y_pos = 0
        self.y_vel = 0
        self.upipe_x = 0
        self.upipe_y = 0
        self.lpipe_y = 0
        self.step_size = step_size
        self.discount = discount
        self.prev_state: List[int] = []
        self.prev_reward: List[float] = []
        self.prev_action: List[int] = []
        self.q_state_action_epsilon = []
        self.make_vectors()
        self.points = 0
        self.hyperparameters = []

    def move(self, y_pos, y_vel, upipes, lpipes, score):
        self.now_ts = time.time_ns() / (10 ** 9)
        self.y_pos = y_pos
        self.y_vel = y_vel
        self.score = score

        d_state = self.now_ts - self.last_state_update_ts
        threshold = t_between_states / self.speedup_factor
        action = False
        if d_state > threshold:
            if upipes[0]['x'] <= self.pos_x:
                self.upipe_x = upipes[0]['x']
                self.upipe_y = upipes[0]['y']
                self.lpipe_y = lpipes[0]['y']
            else:
                self.upipe_x = upipes[1]['x']
                self.upipe_y = upipes[1]['y']
                self.lpipe_y = lpipes[1]['y']

            self.last_state_update_ts = self.now_ts
            self.breakpoint_count += 1
            self.breakpoint_count %= states_between_breakpoints
            action = self.sarsa_onpolicy_tdn_full()

        return flap if action else no_flap

    def sarsa_onpolicy_tdn_full(self) -> bool:
        # State
        state_t1 = self.compute_state(as_index=True)
        # Action
        q_noflap = self.q_state_action_epsilon[state_t1][0]
        q_flap = self.q_state_action_epsilon[state_t1][1]
        epsilon = self.q_state_action_epsilon[state_t1][2]
        # self.q_state_action_epsilon[state_t1][2] /= 1.000000000001
        action_t1 = int(epsilon_greedy(q_noflap=q_noflap, q_flap=q_flap, epsilon=epsilon/(self.score+1)))
         # Reward
        reward_t1 = self.compute_reward()

        self.prev_state.append(state_t1)
        self.prev_action.append(action_t1)
        self.prev_reward.append(reward_t1)

        self.breakpoint_count += 1
        self.breakpoint_count %= states_between_breakpoints
        if self.breakpoint_count == 0:
            print(f'Y_POS, Y_VEL, DX, C_PIPE = {self.compute_state()}', end=" ")
            print(f'Reward = {reward_t1}\n')



        if len(self.prev_state) > self.N:
            state_tn = self.prev_state.pop(0)
            action_tn = self.prev_action.pop(0)
            reward_tn = self.prev_reward.pop(0)
            Qn = self.q_state_action_epsilon[state_tn][action_tn]

            # Compute the discounted sum of n rewards
            G_terms = [self.prev_reward[k] * self.discount**k for k in range(self.N)]
            G_tn = sum(G_terms)
            Qn += self.step_size * (G_tn - Qn)

            self.q_state_action_epsilon[state_tn][action_tn] = Qn

        return bool(action_t1)

    def compute_state(self, as_index=False):
        try:
            Y_POS = map_bin(
                x=self.y_pos,
                minimum=self.ypos_min,
                maximum=self.ypos_max,
                n_bins=NUM_Y_STATES,
                enforce_bounds=False
            )

            Y_VEL = map_bin(
                x=self.y_vel,
                minimum=self.yvel_min,
                maximum=self.yvel_max,
                n_bins=NUM_V_STATES
            )

            DX = map_bin(
                x=self.upipe_x,
                minimum=self.pos_x,
                maximum=self.dx_max - self.pos_x,
                n_bins=NUM_DX_STATES,
                enforce_bounds=False
            )

            C_PIPE = map_bin(
                x=(self.upipe_y + self.pipe_height + self.lpipe_y) / 2,
                minimum=self.pipec_min,
                maximum=self.pipec_max,
                n_bins=NUM_PIPE_STATES
            )

        except ValueError as e:
            print(e.with_traceback())
            raise ValueError
        if as_index:
            index = 0
            index += C_PIPE
            index += DX * (NUM_PIPE_STATES)
            index += Y_VEL * (NUM_DX_STATES * NUM_PIPE_STATES)
            index += Y_POS * (NUM_V_STATES * NUM_DX_STATES * NUM_PIPE_STATES)
            return index
        else:
            return Y_POS, Y_VEL, DX, C_PIPE

    def compute_reward(self):
        #return 1
        dx = (self.upipe_y + self.pipe_height + self.lpipe_y) / 2 - self.y_pos
        return 100000*self.points-math.sqrt(abs(dx)) ** 3

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
        self.episode.append(score)

        # Adjust values on remaining states
        num_states = len(self.prev_state)
        # Losing resulted in the lowest reward
        min_reward = self.q_state_action_epsilon.min()
        self.prev_reward.append(min_reward)
        self.prev_reward.pop(0) # remove R_1

        discount = self.discount
        R_t = 0
        G_t = 0
        for t in range(num_states-1, -1, -1):
            action = self.prev_action.pop(t)
            state = self.prev_state.pop(t)
            R_t = self.prev_reward.pop(t)  # return from Q_
            G_t = R_t + discount * G_t     # discounted return

            Qt = self.q_state_action_epsilon[state][action]
            Qt += self.step_size * (G_t - Qt)
            self.q_state_action_epsilon[state][action] = Qt

    def make_vectors(self):
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
            self.discount, self.step_size, self.N, NUM_Y_STATES, NUM_V_STATES,
            NUM_DX_STATES, NUM_PIPE_STATES, NUM_STATES, EPSILON_START,
            t_between_states
        ]
        self.q_state_action_epsilon = np.insert(
            arr=np.random.normal(loc=1, scale=1, size=(NUM_STATES, 2)),
            obj=2,
            values=np.ones(NUM_STATES) * EPSILON_START,  # epsilon
            axis=1)


def map_bin(x: float, minimum: float, maximum: float, n_bins: int,
            f=lambda x: x, one_indexed=False, enforce_bounds=True):
    if minimum >= maximum:
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
    # returns True for flap and False for no_flap
    if abs(q_flap - q_noflap) < 0.001:
        # if neither are greedy, pick random
        return random.uniform(0, 1) < .5

    # determine greedy state and return it with p = 1 - epsilon
    greedy = q_flap > q_noflap
    if random.uniform(0, 1) > epsilon:
        return greedy
    else:
        return not greedy

