import math
import random
import time

import numpy.random
import pygame
import numpy as np
from typing import List, Tuple

VERBOSE = True
DRAWPOINTS = True
READ_CACHE = True


t_epsilon                   = 0.001     # sec
t_between_states            = 0.10      # sec
states_between_breakpoints  = 25

# hyperparameters
NUM_Y_STATES   = 10  # encodes height of player (FlappyBird expects this to be odd for maintaining center)
NUM_V_STATES   = 8  # encodes player velocity
NUM_DX_STATES  = 5  # encodes distance from pipe to player
NUM_PIPE_STATES = 1  # encodes center position between pipes

no_flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)


class TD1Sarsa:
    def __init__(self, speedup_factor, xpos=57, ypos_min=0, ypos_max=100,
                 yvel_min=-10, yvel_max=10, dx_max=50, pipec_min=0, pipec_max=100,
                 step_size=0.4, discount =0.3):
        self.now_ts                     = time.time_ns() / (10 ** 9)
        self.breakpoint_count           = 0
        self.speedup_factor             = speedup_factor
        self.last_state_update_ts       = self.now_ts
        self.last_flap                  = self.now_ts
        self.episode                    = []
        self.pos_x                      = xpos
        self.ypos_min                   = ypos_min
        self.ypos_max                   = ypos_max
        self.yvel_min                   = yvel_min
        self.yvel_max                   = yvel_max
        self.dx_max                     = dx_max
        self.pipec_min                  = pipec_min
        self.pipec_max                  = pipec_max
        self.pipe_height                = 320
        self.y_pos                      = 0
        self.y_vel                      = 0
        self.upipe_x                    = 0
        self.upipe_y                    = 0
        self.lpipe_y                    = 0
        self.step_size                  = step_size
        self.discount                   = discount
        self.prev_state                 = None
        self.prev_reward                = None
        self.prev_action                = None
        self.this_state                 = None
        self.this_reward                = None
        self.this_action                = None
        self.num_states = NUM_Y_STATES * NUM_V_STATES * NUM_DX_STATES * NUM_PIPE_STATES
        self.q_state_action_epsilon     = []
        self.make_vectors()

    def move(self, y_pos, y_vel, upipes, lpipes, score):
        self.now_ts = time.time_ns() / (10 ** 9)
        self.y_pos = y_pos
        self.y_vel = y_vel

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
            action = self.sarsa_onpolicy_td1()

        return flap if action else no_flap

    def sarsa_onpolicy_td1(self) -> bool:
        self.prev_state = self.this_state
        self.prev_action = self.this_action
        self.prev_reward = self.this_reward

        self.this_state = self.compute_state(as_index=True)
        q_noflap = self.q_state_action_epsilon[self.this_state][0]
        q_flap   = self.q_state_action_epsilon[self.this_state][1]
        epsilon = self.q_state_action_epsilon[self.this_state][2]
        self.q_state_action_epsilon[self.this_state][2] /= 1.1
        print(f'{self.compute_state()} ({self.this_state}) |--> (q_noflap:{q_noflap}, q_flap:{q_flap}, epsilon: {epsilon})')
        self.this_action = int(epsilon_greedy(q_noflap=q_noflap, q_flap=q_flap, epsilon=epsilon))
        self.this_reward = self.compute_reward()

        if self.prev_state is not None:
            if self.prev_action == 2 or self.this_action == 2:
                raise ValueError
            q_prev = self.q_state_action_epsilon[self.prev_state][self.prev_action]
            q_this = self.q_state_action_epsilon[self.this_state][self.this_action]
            q_update = q_prev + self.step_size * (self.prev_reward + self.discount * q_this - q_prev)
            self.q_state_action_epsilon[self.prev_state][self.prev_action] = q_update

        return bool(self.this_action)

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
                x=(self.upipe_y + self.pipe_height + self.lpipe_y)/2,
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
        dx = (self.upipe_y + self.lpipe_y)/2 - self.y_pos
        return -math.sqrt(abs(dx))**3
    


    def save(self):
        from datetime import datetime
        today = datetime.now()
        d1 = today.strftime("%m-%d-%Y %H.%M.%S")
        with open(f"td1/flappy_sarsa-{d1}", 'wb') as f:
            np.save(f, self.q_state_action_epsilon, allow_pickle=True)
            np.save(f, self.episode, allow_pickle=True)
        print(self.q_state_action_epsilon)
        print(self.episode)

    def gameover(self,score):
       self.episode.append(score)

    def make_vectors(self):
        if READ_CACHE:
            try:
                from os import listdir
                names = listdir("td1") # read most recent list
                name = names[len(names)-1]
                with open(f"td1/{name}", 'rb') as f:
                    self.q_state_action_epsilon = np.load(f, allow_pickle=True)
                    self.episode = list(np.load(f, allow_pickle=True))

                print(self.q_state_action_epsilon)
                print(self.episode)
                return
            except:
                pass

        #random
        self.q_state_action_epsilon = np.insert(
            arr=np.random.uniform(low=-2000.0, high=-200.0, size=(self.num_states, 2)),
            obj=2,
            values=np.ones(self.num_states) * 0.5,  # epsilon
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

