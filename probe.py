import numpy as np
from abc import abstractmethod
import gym


class Probe:
    """
    Base class of stimulation electrodes. Translates actions to stimulation sequences.
    """

    def __init__(self, obs_steps, **probe_args):
        self.obs_steps = obs_steps
        self.obs_space = None
        self.prb = None  # set by the neural model upon insertion, type nengo.Probe

    def attach(self, site):  # called by neural model while building the model
        pass # TODO

    def probe(self, data):  # filters the data
        return data


# TODO
# class ProbePak(Probe):
#     """
#     Bundles stimulation sites together to merge their action spaces.
#     TODO only works for discrete xor 1D box action spaces (with same low, high) for now
#     """
#
#     def __init__(self, stim_steps, stims):
#         super(ProbePak).__init__(stim_steps)
#         self.stims = stims
#
#         action_spaces = [stim.action_space for stim in stims]
#         a = action_spaces[0]  # example action
#         if type(a) == gym.spaces.Box:
#             self.action_shapes = [s.shape[0] for s in action_spaces]
#             self.action_space = gym.spaces.Box(a.low, a.high, (np.sum(self.action_shapes),), a.dtype)
#         elif type(a) == gym.spaces.Discrete:
#             self.action_shapes = [s.n for s in action_spaces]
#             self.action_space = gym.spaces.Discrete(np.sum(self.action_shapes))
#         else:
#             raise NotImplemented('discrete or box')
#
#     def stim(self, action):
#         i = 0
#         actions = []
#         for shape in self.action_shapes:
#             actions.append(action[i:i + shape])
#             i += shape
#
#         return {stim.id: stim.stim(a) for a, stim in zip(actions, self.stims)}
