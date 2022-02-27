import time
import pygame

t_epsilon             = 0.001 # sec
time_between_moves    = 0.1 # sec
flaps_per_second      = 0.62
do_nothing = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)

class Agent:
    def __init__(self, speedupfactor):
        self.speedupfactor = speedupfactor
        self.last_move_ts = time.time_ns() / (10 ** 9)
        self.last_flap = self.last_move_ts
        self.y_hist = [(0,0), (0,0), (0,0)]
        
    def flap(self, now_ts):
        dflap = (now_ts - self.last_flap)
        flap_freq = 1.0 / (flaps_per_second)

        if dflap * flap_freq > 1:
            self.last_flap = now_ts
            return True
        else:
            return False

    def move(self, ypos):
        ts = time.time_ns() / (10 ** 9) # accurate clock
        self.y_hist.pop(0)
        self.y_hist.append((ts, ypos))
        dt = self.speedupfactor * (ts - self.last_move_ts)
        # prevents unwanted moves on every function invokation
        if dt > time_between_moves and self.flap(ts):
            self.last_move_ts = ts
            return flap
        else:
            return do_nothing