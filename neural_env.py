import gym
import random


class NeuralEnv(gym.Env):
    """
    Given the bio neural model and the stimpak, it creates a brain stimulation environment
    that can be trained using reinforcement learning methods.
    """

    def __init__(self, neural_model, stimpak, probes):
        super(NeuralEnv, self).__init__()
        # self.__version__ = "0.1.0"

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed):
        random.seed(seed)
