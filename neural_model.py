from abc import abstractmethod
import nengo
import numpy as np


class NeuralModel:
    """
    Base class for biologically plausible brain models.
    """

    def __init__(self, ep_steps, stim_steps, data_path, minibatch_size=1):
        self.net = None
        self.sim = None
        self.ep_steps = ep_steps  # episode length
        self.stim_steps = stim_steps  # stim length, unit of simulation
        self.data_path = data_path
        self.minibatch_size = minibatch_size

        # TODO set to one; only reason to have >1 is when multiple agents run on the same model
        if minibatch_size != 1:
            raise NotImplemented('only minibatch size of 1 allowed')

    @abstractmethod
    def architecture(self):  # run by experiment
        pass

    @abstractmethod
    def build(self, stims, probes):  # run by experiment
        pass  # TODO return input, nengo stim, and probe objects

    @abstractmethod
    def train(self, model_path, load_prev=True, retrain=False):  # run by experiment
        pass

    @abstractmethod
    def test(self, data):  # run by experiment
        pass

    def simulate(self, input):
        # input includes stimulation
        self.sim.run_steps(self.stim_steps, data=input, profile=False, progress_bar=False)

    def probe(self, probes, batch_i=0):
        return [probe.probe(self.sim.data[probe.id][batch_i]) for probe in probes]

    @abstractmethod
    def reset(self):
        pass

    # @staticmethod
    # def attach_stim(stim, x, conn=None):
    #     if conn is None:  # n-to-n
    #         return [nengo.Connection(stim, x)]
    #
    #     connections = []
    #     for stim_i, x_i in zip(conn[0], conn[1]):  # 1-to-1
    #         x_ii = [x_i] if type(x_i) == np.int64 else x_i  # array aloud
    #         for i in x_ii:
    #             connections.append(nengo.Connection(stim[stim_i], x[i]))
    #
    #     return connections
