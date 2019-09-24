import warnings

import gzip
import pickle
from urllib.request import urlretrieve
import zipfile

import nengo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import nengo_dl


warnings.filterwarnings('ignore')
# urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
with gzip.open("mnist.pkl.gz") as f:
    train_data, _, test_data = pickle.load(f, encoding="latin1")
train_data = list(train_data)
test_data = list(test_data)
for data in (train_data, test_data):
    one_hot = np.zeros((data[0].shape[0], 10))
    one_hot[np.arange(data[0].shape[0]), data[1]] = 1
    data[1] = one_hot

# for i in range(3):
#     plt.figure()
#     plt.imshow(np.reshape(train_data[0][i], (28, 28)), cmap="gray")
#     # plt.axis('off')
#     plt.title(str(np.argmax(train_data[1][i])))
#     plt.show()


def attach_stim(stim, x, conn=None):
    if conn is None:  # n-to-n
        return [nengo.Connection(stim, x)]

    connections = []
    for stim_i, x_i in zip(conn[0], conn[1]):
        x_ii = [x_i] if type(x_i) == np.int64 else x_i  # array aloud
        for i in x_ii:
            connections.append(nengo.Connection(stim[stim_i], x[i]))

    return connections


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
    inp = nengo.Node([0] * 28 * 28)


    # fully connected encoder
    x = nengo_dl.tensor_layer(inp, tf.layers.dense, shape_in=(28 * 28,), units=32)  # TODO try less parameters
    x = nengo_dl.tensor_layer(x, neuron_type)

    x = nengo_dl.tensor_layer(x, tf.layers.dense, shape_in=(32,), units=16)
    x = nengo_dl.tensor_layer(x, neuron_type)

    x = nengo_dl.tensor_layer(x, tf.layers.dense, shape_in=(16,), units=8)
    latent = nengo_dl.tensor_layer(x, neuron_type)

    # fully connected decoder
    x = nengo_dl.tensor_layer(latent, tf.layers.dense, shape_in=(8,), units=16)
    x = nengo_dl.tensor_layer(x, neuron_type)

    x = nengo_dl.tensor_layer(x, tf.layers.dense, shape_in=(16,), units=32)
    x = nengo_dl.tensor_layer(x, neuron_type)

    # x = nengo_dl.tensor_layer(x, tf.layers.dense, shape_in=(32,), units=64)
    # x = nengo_dl.tensor_layer(x, neuron_type)

    x = nengo_dl.tensor_layer(x, tf.layers.dense, shape_in=(32,), units=28 * 28)
    # x = nengo_dl.tensor_layer(x, neuron_type)


    # # ENCODER
    # # add the first convolutional layer
    # x = nengo_dl.tensor_layer(inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=16, kernel_size=3, padding='same')
    #
    # # apply the neural nonlinearity
    # x = nengo_dl.tensor_layer(x, neuron_type)
    #
    # # add another convolutional layer
    # x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(28, 28, 16), filters=16, kernel_size=3, padding='same')
    # x = nengo_dl.tensor_layer(x, neuron_type)
    #
    # # add a pooling layer
    # x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(28, 28, 16), filters=32, kernel_size=3, strides=2, padding='same')
    # # x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d, shape_in=(24, 24, 64), pool_size=2, strides=2)
    #
    # # another convolutional layer
    # # (W - Fw + 2P) / Sw + 1
    # x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(14, 14, 32), filters=32, kernel_size=3, padding='same')
    # x = nengo_dl.tensor_layer(x, neuron_type)
    #
    # # another pooling layer -> 7x7x128
    # x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(14, 14, 32), filters=64, kernel_size=3, strides=2, padding='same')
    # # x = nengo_dl.tensor_layer(x, tf.layers.average_pooling2d, shape_in=(10, 10, 128), pool_size=2, strides=2)
    #
    # # latent
    # latent = x
    #
    # # DECODER
    # x = nengo_dl.tensor_layer(x, tf.layers.conv2d_transpose, shape_in=(7, 7, 64), filters=32, kernel_size=3, strides=2, padding='same')
    # x = nengo_dl.tensor_layer(x, neuron_type)
    # x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(14, 14, 32), filters=32, kernel_size=3, padding='same')
    # x = nengo_dl.tensor_layer(x, tf.layers.conv2d_transpose, shape_in=(14, 14, 32), filters=16, kernel_size=3, strides=2, padding='same')
    # x = nengo_dl.tensor_layer(x, neuron_type)
    # x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(28, 28, 16), filters=16, kernel_size=3, padding='same')
    # x = nengo_dl.tensor_layer(x, neuron_type)
    # x = nengo_dl.tensor_layer(x, tf.layers.conv2d, shape_in=(28, 28, 16), filters=1, kernel_size=1, padding='same')

    # linear readout
    # x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)
    # x = nengo_dl.tensor_layer(x, tf.identity)

    # x = x + out_stim
    out_stim = nengo.Node([0] * 10)
    # stim_conns = attach_stim(out_stim, x, (np.arange(10), np.arange(15)))

    # we'll create two different output probes, one with a filter
    # (for when we're simulating the network over time and
    # accumulating spikes), and one without (for when we're
    # training the network using a rate-based approximation)
    out_p = nengo.Probe(x)
    out_p_filt = nengo.Probe(x, synapse=0.1)


minibatch_size = 200
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)
# sim.freeze_params(stim_conns)

# add the single timestep to the training data
# black = np.zeros(train_data[0][:10000].shape)
# train_data[0] = np.concatenate([black, train_data[0]], axis=0)
# np.random.shuffle(train_data[0])
train_data = {inp: train_data[0][:, None, :],
              out_p: train_data[0][:, None, :]}

# when testing our network with spiking neurons we will need to run it
# over time, so we repeat the input/target data for a number of
# timesteps. we're also going to reduce the number of test images, just
# to speed up this example.
n_steps = 30
test_data = {
    inp: np.tile(test_data[0][:minibatch_size*2, None, :],
                 (1, n_steps, 1)),
    out_p_filt: np.tile(test_data[0][:minibatch_size*2, None, :],
                        (1, n_steps, 1))}


def objective(outputs, targets):
    # return tf.nn.softmax_cross_entropy_with_logits_v2(
    #     logits=outputs, labels=targets)
    return tf.losses.mean_squared_error(labels=targets, predictions=outputs)


opt = tf.train.RMSPropOptimizer(learning_rate=0.001)


def classification_error(outputs, targets):
    # return 100 * tf.reduce_mean(
    #     tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
    #                          tf.argmax(targets[:, -1], axis=-1)),
    #             tf.float32))
    return tf.losses.mean_squared_error(labels=targets, predictions=outputs)


# print("error before training: %.2f%%" % sim.loss(
#     test_data, {out_p_filt: classification_error}))

load_prev = False
do_training = True
nepochs = 10
if load_prev:
    sim.load_params("./models/mnist_auto_params")
    print('PARAMETERS LOADED')

if do_training:
    # run training
    sim.train(train_data, opt, objective={out_p: objective}, n_epochs=nepochs)

    # save the parameters to file
    sim.save_params("./models/mnist_auto_params")


print("error after training: %.2f%%" % sim.loss(
    test_data, {out_p_filt: classification_error}))


# simulate, read probes, activate neuron zero, repeat
# out_stim_patterns = np.ones((minibatch_size, n_steps, 10))
out_stim_patterns = np.zeros((minibatch_size, n_steps, 10))
out_stim_patterns[:, :, 0] = 100
# out_stim_patterns = np.random.randint(0, 2, (minibatch_size, n_steps, 10))

# sim.freeze_params([stim_to_out, x_to_out])
# test_data[inp][::2] = np.zeros(test_data[inp][0].shape)  # black out every second one
sim.run_steps(n_steps, data={inp: test_data[inp][:minibatch_size], out_stim: out_stim_patterns})

for i in range(20):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.reshape(test_data[inp][i, 0], (28, 28)), cmap="gray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.reshape(sim.data[out_p_filt][i, -1], (28, 28)), cmap="gray")
    plt.legend([str(i) for i in range(10)], loc="upper left")
    plt.xlabel("time")
    plt.show()

sim.close()
