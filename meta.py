# LAYOUT SETTINGS
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 175 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79

# AGENT SETTINGS
T_BETWEEN_STATES = .1
CHECKPOINT_DT   = 5 * 60     # every 5 minutes
X_POS_AGENT  = 57
Y_MIN_AGENT  = 0                    # top of screen (0 = top)
Y_MAX_AGENT  = BASEY                # where player crashes into ground
Y_MAX_VELOCITY = 10
Y_MIN_VELOCITY = -9


# PIPE SETTINGS
PERCENT_TOGETHER = 5
PIPEHEIGHT = 320
# PIPEGAPSIZE  = 100
# Y_MIN_LPIPE  = int(Y_MAX_AGENT * 0.2)     # smallest y-coordinate of lower pipe opening
# Y_MAX_LPIPE  = int(Y_MAX_AGENT * 0.8)     # largest y-coordinate of lower pipe opening
PIPEGAPSIZE  = BASEY * (1 - PERCENT_TOGETHER/200)
Y_MAX_LPIPE = BASEY
Y_MIN_LPIPE = 0
X_MAX_PIPE   = SCREENWIDTH - X_POS_AGENT



# Checkpointing and debugging
RESET_EPSILON_GREEDY = False
STATES_BETWEEN_LOG = 1
LOG = True