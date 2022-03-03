import pygame
import time

# Gameplay Attributes
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 100
Y_MIN_AGENT  = 0                    # top of screen (0 = top)
Y_MAX_AGENT  = SCREENHEIGHT * 0.79  # where player crashes into ground
Y_MIN_LPIPE  = 0                    # smallest y-coordinate of lower pipe opening
Y_MAX_LPIPE  = 0                    # largest y-coordinate of lower pipe opening


# (Discretized) State Attributes
NUM_Y_STATES = 10                   # encodes height of player (this should be odd for keeping in center)
NUM_V_STATES = 10                   # encodes player velocity
NUM_DX_STATES = 6                   # encodes distance from pipe to player
NUM_PIPE_STATES = 8                 # encodes center position between pipes
NUM_STATES = NUM_Y_STATES * NUM_V_STATES * NUM_DX_STATES * NUM_PIPE_STATES


# Training Hyperparameters
TRAINING_ALGORITHM = 0

# Checkpointing and debugging
CHECKPOINT_DT   = 5 * 60     # every 5 minutes


no_flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
flap = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)


class Agent:
    def __init__(self, speedup_factor=1, x_pos=57):
        self.now_ts = time.time_ns() / (10 ** 9)    # used to discretize game
        self.last_state_update = self.now_ts        # used to discretize game
        self.episodes = [] # (num                         # used for gathering data
        speedup_factor = speedup_factor


