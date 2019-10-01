import numpy as np
from abc import abstractmethod
import gym
import nengo


class Stim:
    """
    Base class of stimulation electrodes. Translates actions to stimulation sequences.
    By placing the same stim object at multiple sites, one stim can represent a nengo node of size > 1.
    By default it will create 1-to-1 connections if the number of placements == n, but by providing i to attach(),
    the connections may be constructed arbitrary.
    TODO for now only works with nengo
    """

    def __init__(self, n, stim_steps, **node_args):
        self.n = n  # number of electrodes
        self.stim_steps = stim_steps
        self.action_space = None
        self.node = None  # set by the neural model upon attachment, type nengo.Node
        self.node_args = node_args
        self.attach_i = 0  # index of the neuron within the nengo node; enables 1-to-1 connections

    def attach(self, site, i=None):  # called by neural model while building the model
        if self.node is None:
            self.node = nengo.Node(self.n, self.node_args)

        if i is None:  # i overwrites attach_i
            i = self.attach_i
        self.attach_i = (self.attach_i + 1) % self.n
        return nengo.Connection(self.node[i], site)

    @abstractmethod
    def stim(self, action):
        pass


class StimPak(Stim):
    """
    Bundles stimulation sites together to merge their action spaces.
    TODO only works for discrete xor 1D box action spaces (with same low, high) for now
    """

    def __init__(self, stim_steps, stims):
        super(StimPak).__init__(stim_steps)
        self.stims = stims

        action_spaces = [stim.action_space for stim in stims]
        a = action_spaces[0]  # example action
        if type(a) == gym.spaces.Box:
            self.action_shapes = [s.shape[0] for s in action_spaces]
            self.action_space = gym.spaces.Box(a.low, a.high, (np.sum(self.action_shapes),), a.dtype)
        elif type(a) == gym.spaces.Discrete:
            self.action_shapes = [s.n for s in action_spaces]
            self.action_space = gym.spaces.Discrete(np.sum(self.action_shapes))
        else:
            raise NotImplemented('discrete or box')

    def stim(self, action):
        i = 0
        actions = []
        for shape in self.action_shapes:
            actions.append(action[i:i + shape])
            i += shape

        return {stim.id: stim.stim(a) for a, stim in zip(actions, self.stims)}
