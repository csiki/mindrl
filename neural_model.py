from abc import abstractmethod


class NeuralModel:
    """
    Base class for biologically plausible brain models.
    """

    def __init__(self):
        # self.net
        pass

    @abstractmethod
    def build(self, stims, probes):
        pass

    @abstractmethod
    def architecture(self):
        pass

    # @abstractmethod
    def sim(self, stimulation):
        pass

    # @abstractmethod
    def probe(self, sites):
        pass

    # @abstractmethod
    def train(self, data):
        pass

    def test(self, data):
        pass
