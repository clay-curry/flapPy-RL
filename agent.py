import multiprocessing
from multiprocessing import Process
from pygame import KEYDOWN


class Agent(Process):
    def __init__(self):
        Process.__init__(self)


    def flap(self):
