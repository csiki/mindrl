import gym
import random
from stim import StimPak


class NeuralEnv(gym.Env):
    """
    Given the bio neural model and the stimpak, it creates a brain stimulation environment
    that can be trained using reinforcement learning methods.
    """

    def __init__(self, neural_model, stims, probes, ep_steps, stim_steps):
        super(NeuralEnv, self).__init__()
        self.model = neural_model
        self.stims = stims
        self.probes = probes
        self.ep_steps = ep_steps
        self.stim_steps = stim_steps

        # simulation
        self.model.reset()

        # action space
        self.stimpak = StimPak(stim_steps, stims)
        self.action_space = self.stimpak.action_space

        # observation space
        self.probepak

        self.curr_step = -1
        self.curr_episode = -1
        self.action_episode_memory = []

    def step(self, action):
        if self.curr_step >= self.ep_steps:
            raise RuntimeError("episode is done")
        self.curr_step += 1

    def reset(self):
        self.model.reset()

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed):
        random.seed(seed)
