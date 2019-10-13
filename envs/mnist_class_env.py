import neural_env
import gym
import random
import numpy as np
import logging.config
import nengo
import nengo_dl
import tensorflow as tf
import os
import warnings
import gzip
import pickle
from urllib.request import urlretrieve
import zipfile
from pprint import pprint
import neural_env
import tensorflow as tf
import matplotlib.pyplot as plt


def attach_stim(stim, x, conn=None):
    if conn is None:  # n-to-n
        return [nengo.Connection(stim, x)]

    connections = []
    for stim_i, x_i in zip(conn[0], conn[1]):
        x_ii = [x_i] if type(x_i) == np.int64 else x_i  # array aloud
        for i in x_ii:
            connections.append(nengo.Connection(stim[stim_i], x[i]))

    return connections


class MNISTClassEnv(gym.Env):
    def __init__(self, desired_outputs=[0]):
        super(MNISTClassEnv, self).__init__()
        self.__version__ = "0.1.0"
        logging.info("MNIST Classification Brain - Version {}".format(self.__version__))

        # model specific vars
        self.potential_desired_outputs = desired_outputs
        self.desired_output = desired_outputs[0]  # initial
        self.desired_output_onehot = np.zeros(10, dtype=np.float32)
        self.desired_output_onehot[self.desired_output] = 1.
        self.n_steps = 30
        self.stim_steps = 5  # 1 won't work, 5 and 10 do
        self.idle_steps = self.n_steps * 2  # * n_steps amount of steps idling
        self.ep_len = self.n_steps // self.stim_steps
        self.output = np.zeros(10)
        self.output_norm = np.zeros(10)
        self.action = None
        self.reward = None
        self.action_space_size = 1000
        self.stim = None  # nengo node
        self.stim_amp = 30.

        self.pretraining = False  # train the bio network first
        self.testing = False
        self.minibatch_size = 200 if self.pretraining or self.testing else 1

        # load and net, init sim
        self.net = self._build_net()

        # attach stim
        # hopeless_conn = np.random.choice(self._find_node('pool2').size_in, self.action_space_size - 10)
        # attachments = [(('stim', range(10)), ('output', range(10))),
        #                (('stim', range(10, self.action_space_size)), ('pool2', hopeless_conn))]
        hopeless_conn = np.random.choice(self._find_node('pool2').size_in, self.action_space_size)
        attachments = [(('stim', range(self.action_space_size)), ('pool2', hopeless_conn))]
        print('ATTACHMENTS:', attachments)
        with self.net:
            self.stim_connections = self.add_conn(attachments)

        # data loading
        ntest_imgs = 500
        self.train_data, self.test_data = self._load_data('mnist.pkl.gz')
        self.train_data = {self.inp: self.train_data[0][:, None, :],
                           self.out_p: self.train_data[1][:, None, :]}

        self.stim_steps = self.n_steps if self.pretraining or self.testing else self.stim_steps
        self.test_data = {
            self.inp: np.tile(self.test_data[0][:ntest_imgs, None, :], (1, self.stim_steps, 1)),
            self.out_p_filt: np.tile(self.test_data[1][:ntest_imgs, None, :], (1, self.stim_steps, 1))}
        self.rand_test_data = np.random.choice(self.test_data[self.inp].shape[0], self.minibatch_size)

        # idle state
        self.no_img = np.zeros((self.minibatch_size, self.idle_steps, 28*28))
        self.no_stim = np.zeros((self.minibatch_size, self.idle_steps, self.action_space_size))

        # load or train net
        self.sim = nengo_dl.Simulator(self.net, minibatch_size=self.minibatch_size)
        self._train(retrain=self.pretraining)
        if self.testing:
            # no stim
            print('NO STIM TESTING')
            self._test(np.zeros((self.minibatch_size, self.n_steps, self.action_space_size)))

            # stim
            for stim_site in range(10):
                print('HARDCORE STIM OF SITE {}'.format(stim_site))
                stim_pattern = np.zeros((self.minibatch_size, self.stim_steps, self.action_space_size))
                stim_pattern[:, :, stim_site] = self.stim_amp
                self._test(stim_pattern)
            print('TESTING DONE, RUN THE NETWORK AGAIN TO TRAIN THE RL MODEL')

        # gym specific vars
        # self.TOTAL_TIME_STEPS = 2
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.float32)
        # self.action_space = gym.spaces.Discrete(self.action_space_size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10 + 10,), dtype=np.float32)  # output and onehot desired output
        self.curr_step = -1
        self.curr_episode = -1
        self.action_episode_memory = []

        # rendering stuff
        # plots:
        #   action distribution (each line an action value, across time and episodes)
        #   output norm (across time and episodes)
        self.fig, self.axes = plt.subplots(2)
        self.axes[0].set_title('action')
        self.axes[1].set_title('observation')
        # self.axes_data = [[], []]
        self.render_i = 0
        self.action_plot_only = [0, 8]  # range(10)
        self.obs_plot_only = [0, 8]  # range(10)
        self.axes[0].legend([str(i) for i in self.action_plot_only])
        self.axes[1].legend([str(i) for i in self.obs_plot_only])

    @staticmethod
    def _load_data(data_path='mnist.pkl.gz'):
        warnings.filterwarnings('ignore')
        if not os.path.isfile(data_path):
            urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")

        with gzip.open("mnist.pkl.gz") as f:
            train_data, _, test_data = pickle.load(f, encoding="latin1")

        train_data = list(train_data)
        test_data = list(test_data)
        for data in (train_data, test_data):
            one_hot = np.zeros((data[0].shape[0], 10))
            one_hot[np.arange(data[0].shape[0]), data[1]] = 1
            data[1] = one_hot

        return train_data, test_data

    def _train(self, retrain=False, train_path='./mnist_params', epochs=10):
        if retrain:
            opt = tf.train.RMSPropOptimizer(learning_rate=0.001)
            self.sim.train(self.train_data, opt, objective={self.out_p: self._objective}, n_epochs=epochs)
            self.sim.save_params(train_path)
            print('PRETRAINING DONE, RUN THE NETWORK AGAIN TO TRAIN THE RL MODEL')
            exit(0)

        self.sim.load_params(train_path)
        print('PARAMETERS LOADED')

    def _test(self, stim_pattern):
        from mnist_class import classification_error
        print("error after training: %.2f%%" % self.sim.loss(self.test_data, {self.out_p_filt: classification_error}))
        self.sim.run_steps(self.n_steps, data={self.inp: self.test_data[self.inp][:self.minibatch_size],
                                               self.stim: stim_pattern})

        for i in range(5):
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.reshape(self.test_data[self.inp][i, 0], (28, 28)), cmap="gray")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.plot(self.sim.trange(), self.sim.data[self.out_p_filt][i])
            plt.legend([str(i) for i in range(10)], loc="upper left")
            plt.xlabel("time")

        plt.show()
        self._idle()

    def _build_net(self):
        with nengo.Network() as net:
            # set some default parameters for the neurons that will make
            # the training progress more smoothly
            net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
            net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            neuron_type = nengo.LIF(amplitude=0.01)

            # we'll make all the nengo objects in the network
            # non-trainable. we could train them if we wanted, but they don't
            # add any representational power. note that this doesn't affect
            # the internal components of tensornodes, which will always be
            # trainable or non-trainable depending on the code written in
            # the tensornode.
            nengo_dl.configure_settings(trainable=False)

            # the input node that will be used to feed in input images
            self.inp = nengo.Node([0] * 28 * 28, label='input')

            # add the first convolutional layer
            x = nengo_dl.tensor_layer(self.inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32, kernel_size=3)
            x = nengo_dl.tensor_layer(x, tf.identity)
            x.label = 'conv1'

            # apply the neural nonlinearity
            x = nengo_dl.tensor_layer(x, neuron_type)
            x.label = 'conv1_act'

            # add another convolutional layer
            x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(26, 26, 32), filters=64, kernel_size=3)
            x = nengo_dl.tensor_layer(x, tf.identity)
            x.label = 'conv2'
            x = nengo_dl.tensor_layer(x, neuron_type)
            x.label = 'conv2_act'

            # add a pooling layer
            x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d, shape_in=(24, 24, 64), pool_size=2, strides=2)
            x = nengo_dl.tensor_layer(x, tf.identity)
            x.label = 'pool1'

            # another convolutional layer
            x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(12, 12, 64), filters=128, kernel_size=3)
            x = nengo_dl.tensor_layer(x, tf.identity)
            x.label = 'conv3'
            x = nengo_dl.tensor_layer(x, neuron_type)
            x.label = 'conv3_act'

            # another pooling layer
            x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d, shape_in=(10, 10, 128), pool_size=2, strides=2)
            x = nengo_dl.tensor_layer(x, tf.identity)
            x.label = 'pool2'

            # linear readout
            x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)
            x = nengo_dl.tensor_layer(x, tf.identity)
            x.label = 'output'
            # x = nengo_dl.tensor_layer(x, tf.identity)
            # x.label = 'output_id'

            # x = x + stim
            self.stim = nengo.Node([0] * self.action_space_size, label='stim')
            # self.stim_conns = attach_stim(self.stim, x, (np.arange(10), np.arange(15)))

            # we'll create two different output probes, one with a filter
            # (for when we're simulating the network over time and
            # accumulating spikes), and one without (for when we're
            # training the network using a rate-based approximation)
            self.out_p = nengo.Probe(x)
            self.out_p_filt = nengo.Probe(x, synapse=0.1)

        return net

    def _find_node(self, label):
        for node in self.net.nodes:
            if node.label == label:
                return node
        raise IndexError('node with label "{}" doesn\'t exist'.format(label))

    def add_conn(self, conn, **conn_args):  # typically used for to connect the stimulation to network neurons
        # conn: [( ('stim', [0,1,2]), ('conv1', [1,2,3]) ), ( ('stim', [3,4,5]), ('conv2', [4,42,420]) )]
        # strings are the labels of nengo nodes
        # indices list if empty, then it spawns ensemble lvl connection
        # returns the connections
        connections = []
        for pair in conn:
            src = self._find_node(pair[0][0])
            dst = self._find_node(pair[1][0])
            sis = pair[0][1]
            dis = pair[1][1]
            if len(sis) == 0 and len(dis) == 0:
                connections.append(nengo.Connection(src, dst, **conn_args))
            else:
                for si, di in zip(sis, dis):
                    connections.append(nengo.Connection(src[si], dst[di], **conn_args))

        return connections

    @staticmethod
    def _objective(outputs, targets):
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets)

    @staticmethod
    def _classification_error(outputs, targets):
        return 100 * tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                                    tf.argmax(targets[:, -1], axis=-1)), tf.float32))

    def step(self, action):
        if self.curr_step >= self.ep_len - 1:
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        self._take_action(action)
        self.reward = self._get_reward()
        obs = self._get_state()
        return obs, self.reward, self.curr_step >= self.ep_len - 1, {}

    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)

        stim_pattern = np.zeros((self.minibatch_size, self.stim_steps, self.action_space_size))
        if type(self.action_space) == gym.spaces.Box:
            stim_pattern[:, :, :] = action * self.stim_amp
        elif type(self.action_space) == gym.spaces.Discrete:
            stim_pattern[:, :, action] = self.stim_amp  # discrete space
        else:
            raise NotImplemented('yo, whatyadoin')

        self.sim.run_steps(self.stim_steps, data={self.inp: self.test_data[self.inp][self.rand_test_data],
                                                  self.stim: stim_pattern}, profile=False, progress_bar=False)

        self.action = action
        self.output = self.sim.data[self.out_p_filt][0][-1]

    def _get_reward(self):
        self.output_norm = (self.output - np.min(self.output)) / (np.max(self.output) - np.min(self.output))
        return 2 * self.output_norm[self.desired_output] - np.sum(self.output_norm)

    def _idle(self):  # idle no image and no stim between showing images
        # simulate the network without input (with 0 inputs) to force it back to baseline
        # same amount of steps as for showing an image
        if self.curr_episode % 500 == 0:
            print('IDLING @{}'.format(self.curr_episode))
        self.sim.run_steps(self.idle_steps, data={self.inp: self.no_img, self.stim: self.no_stim},
                           profile=False, progress_bar=False)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.rand_test_data = np.random.choice(self.test_data[self.inp].shape[0], self.minibatch_size)
        self.desired_output = np.random.choice(self.potential_desired_outputs, 1)[0]
        self.desired_output_onehot = np.zeros(10, dtype=np.float32)
        self.desired_output_onehot[self.desired_output] = 1.

        self._idle()
        return self._get_state()

    def render(self, mode='human', close=False):
        r = {'desired': self.desired_output, 'action': self.action, 'output': self.output, 'output_norm': self.output_norm,
             'reward': self.reward, 'random_img': self.rand_test_data[0]}
        if self.curr_step == -1:
            pprint('=================================================')
        pprint('-------------------------------------------------')
        pprint('@{}/{}:'.format(self.curr_episode, self.curr_step))
        pprint(r)

        # plot
        # actions
        if self.action is not None:
            if self.curr_step == -1:
                self.axes[0].plot([self.render_i - .5, self.render_i - .5], [-1, 1], color='red')
            for i in self.action_plot_only:  # range(len(self.action)):
                self.axes[0].scatter(self.render_i, self.action[i], label=str(i), color='C{}'.format(i))
            plt.pause(0.05)
        # normalized outputs
        if self.output_norm is not None:
            if self.curr_step == -1:
                self.axes[1].plot([self.render_i - .5, self.render_i - .5], [.25, .75], color='red')
            for i in self.obs_plot_only:  # range(len(self.output_norm)):
                self.axes[1].scatter(self.render_i, self.output_norm[i], label=str(i), color='C{}'.format(i))
            plt.pause(0.001)

        # self.axes[0].legend()
        # self.axes[1].legend()

        self.render_i += 1

    def _get_state(self):
        return np.concatenate([self.desired_output_onehot, self.output_norm])
        # return np.concatenate(([self.desired_output], self.output_norm))

    def seed(self, seed):
        random.seed(seed)


if __name__ == '__main__':
    pass
