import time
import pygame
import numpy as np

t_epsilon             = 0.0001 # sec
t_between_actions     = 0.02   # sec
default_flaps_per_second      = 0.62
no_flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
DRAWPOINTS = True


class Agent:
    def __init__(self, flaps_per_second = default_flaps_per_second):
        now_ts                  = time.time_ns() / (10 ** 9)
        self.start_ts           = now_ts
        self.last_flap          = now_ts
        self.flaps_per_second   = flaps_per_second
        self.lastpolicycall_ts  = now_ts    # last time policy was invoked
        self.pos_hist           = [(0,0), (0,0), (0,0)] # (t, x, y)
        self.upipepos_hist      = [(0,0), (0,0), (0,0)] # (t, x, y)
        self.lpipepos_hist      = [(0,0), (0,0), (0,0)] # (t, x, y)
        self.n_points           = 0
        

    def move(self, y_pos, u_pos, l_pos, n_points):
        now_ts = time.time_ns() / (10 ** 9)
        self.u_pos = u_pos
        self.l_pos = l_pos 
        policy_dt = now_ts - self.lastpolicycall_ts
        height_dt = now_ts - self.upipepos_hist[2][0]
        self.n_points = n_points

        next_move = no_flap
        if height_dt > (t_between_actions / 4):
            # check if first pipe has moved off screen
            first_pipe_leftx = u_pos[0][0]
            if first_pipe_leftx > self.upipepos_hist[0][0]: 
                self.upipepos_hist.pop(0)
                self.lpipepos_hist.pop(0)

            dx = u_pos[0][0] - self.upipepos_hist[0][0]
            for list in self.upipepos_hist + self.lpipepos_hist:
                print(list)
            
        if policy_dt > t_between_actions:
            next_move = flap if self.policy(now_ts=now_ts) else no_flap
        return next_move

    def policy(self, now_ts) -> bool:
        dflap = (now_ts - self.last_flap)
        flap_freq = 1.0 / (self.flaps_per_second)

        if dflap * flap_freq > 1:
            self.last_flap = now_ts
            return True
        else:
            return False
    
    def draw_points(self, SCREEN):
        uPipes = self.u_pos
        lPipes = self.l_pos
        if DRAWPOINTS:
            # draw obstacles
            for u, l in zip(uPipes, lPipes):
                pygame.draw.circle(SCREEN,(255,0,0),(u[0],u[1]),5.5)
                pygame.draw.circle(SCREEN,(255,0,0),(u[2],u[3]),5)
                pygame.draw.circle(SCREEN,(255,0,0),(l[0],l[1]),5)
                pygame.draw.circle(SCREEN,(255,0,0),(l[2],l[3]),5)
            # draw trajectory
            for t in range(20):
                pass

