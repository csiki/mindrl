import numpy as np
from abc import abstractmethod


class Stim:
    """
    Base class of stimulation electrodes. Translates actions to stimulation sequences.
    """

    def __init__(self):
        pass

    @abstractmethod
    def stim(self, action):
        pass

    @abstractmethod
    def action_space(self):
        pass


class StimPak(Stim):
    """
    Bundles stimulation sites together to computationally accelerate the mapping between the agent's action space
    and the stimulation sequences passed to the bio model.
    """

    def __init__(self, stims):
        super(StimPak).__init__()
    
    def stim(self, action):
        pass

    def action_space(self):
        pass