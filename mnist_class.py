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

    # add the first convolutional layer
    x = nengo_dl.tensor_layer(
        inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
        kernel_size=3)

    # apply the neural nonlinearity
    x = nengo_dl.tensor_layer(x, neuron_type)

    # add another convolutional layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.conv2d, shape_in=(26, 26, 32),
        filters=64, kernel_size=3)
    x = nengo_dl.tensor_layer(x, neuron_type)

    # add a pooling layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.average_pooling2d, shape_in=(24, 24, 64),
        pool_size=2, strides=2)

    # another convolutional layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.conv2d, shape_in=(12, 12, 64),
        filters=128, kernel_size=3)
    x = nengo_dl.tensor_layer(x, neuron_type)

    # another pooling layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.average_pooling2d, shape_in=(10, 10, 128),
        pool_size=2, strides=2)

    # linear readout
    x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)

    # x = x + out_stim
    out_stim = nengo.Node([0] * 10)
    # out_stim = nengo_dl.tensor_layer(out_stim, tf.identity)
    # outen = nengo_dl.tensor_layer(out_stim + x, tf.identity)
    outen = nengo_dl.TensorNode(lambda t, x: tf.identity(x), size_in=10)
    stim_to_out = nengo.Connection(out_stim, outen, synapse=None, transform=np.eye(10))
    x_to_out = nengo.Connection(x, outen, synapse=None, transform=np.eye(10))
    # TODO FIXME tried freezing the fuckers, still not working

    # outen = nengo.Ensemble(1000, 10)
    # nengo.Connection(x, outen)
    # nengo.Connection(x, outen, transform=np.eye(10))
    x = outen

    # we'll create two different output probes, one with a filter
    # (for when we're simulating the network over time and
    # accumulating spikes), and one without (for when we're
    # training the network using a rate-based approximation)
    out_p = nengo.Probe(x)
    out_p_filt = nengo.Probe(x, synapse=0.1)


minibatch_size = 200
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)
sim.freeze_params([stim_to_out, x_to_out])

# add the single timestep to the training data
train_data = {inp: train_data[0][:, None, :],
              out_p: train_data[1][:, None, :]}

# when testing our network with spiking neurons we will need to run it
# over time, so we repeat the input/target data for a number of
# timesteps. we're also going to reduce the number of test images, just
# to speed up this example.
n_steps = 30
test_data = {
    inp: np.tile(test_data[0][:minibatch_size*2, None, :],
                 (1, n_steps, 1)),
    out_p_filt: np.tile(test_data[1][:minibatch_size*2, None, :],
                        (1, n_steps, 1))}


def objective(outputs, targets):
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=outputs, labels=targets)


opt = tf.train.RMSPropOptimizer(learning_rate=0.001)


def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


# print("error before training: %.2f%%" % sim.loss(
#     test_data, {out_p_filt: classification_error}))

do_training = False
if do_training:
    # run training
    sim.train(train_data, opt, objective={out_p: objective}, n_epochs=10)

    # save the parameters to file
    sim.save_params("./mnist_params")
else:
    # download pretrained weights
    # urlretrieve("https://drive.google.com/uc?export=download&id=1u9JyNuRxQDUcFgkRnI1qfJVFMdnGRsjI", "mnist_params.zip")
    # with zipfile.ZipFile("mnist_params.zip") as f:
    #     f.extractall()

    # load parameters
    sim.load_params("./mnist_params")
    print('PARAMETERS LOADED')


print("error after training: %.2f%%" % sim.loss(
    test_data, {out_p_filt: classification_error}))


# simulate, read probes, activate neuron zero, repeat
# out_stim_patterns = np.ones((minibatch_size, n_steps, 10))
out_stim_patterns = np.zeros((minibatch_size, n_steps, 10))
out_stim_patterns[:, :, 0] = 1
# out_stim_patterns = np.random.randint(0, 2, (minibatch_size, n_steps, 10))

sim.freeze_params([stim_to_out, x_to_out])
sim.run_steps(n_steps, data={inp: test_data[inp][:minibatch_size], out_stim: out_stim_patterns})

for i in range(5):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.reshape(test_data[inp][i, 0], (28, 28)), cmap="gray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(sim.trange(), sim.data[out_p_filt][i])
    plt.legend([str(i) for i in range(10)], loc="upper left")
    plt.xlabel("time")
    plt.show()

sim.close()
