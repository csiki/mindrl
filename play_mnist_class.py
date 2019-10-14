import gym
import envs
from spinup.utils.run_utils import ExperimentGrid
from spinup import ddpg, ppo, sac, trpo, td3
import tensorflow as tf
import numpy as np
import os


def select_gpu(gpu_id=-1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id != -1 else '0,1'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print('GPU FOUND:', gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    print('RUNNING ON GPU #{}'.format(gpu_id))


# TODO try other rl models
# TODO add image input and other neuron states as state - image input would likely need a conv net built in the agent
# TODO build your own graph network mlp to be passed to the agent
# TODO implement stimulation on multiple images at the same time = minibatch size > 1, multiple agents sync stim 1 model
# TODO add probe to attachment sites in general, not just output

# from https://gist.github.com/krishpop/f4b2aa8d60d7b22bce8c258fd68ab11c
def run_experiment(args, rl_model):
    def env_fn():
        import envs  # registers custom envs to gym env registry
        return gym.make(args.env_name, desired_outputs=args.desired_outputs)

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('env_fn', env_fn)
    # eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 100)
    eg.add('steps_per_epoch', 6*500)  # FIXME
    eg.add('save_freq', 10)
    # eg.add('num_runs', args.num_runs)
    eg.add('max_ep_len', 6)  # FIXME get it from env

    # ppo
    eg.add('pi_lr', 3e-3)

    # actor-critic
    eg.add('ac_kwargs:activation', tf.tanh, '')
    # eg.add('ac_kwargs:hidden_sizes', [32, 32])
    eg.run(rl_model, num_cpu=args.cpu, data_dir=args.data_dir)


if __name__ == '__main__':
    select_gpu(0)

    env_fn = lambda: gym.make('MNISTClassEnv-v0')

    # trpo(env_fn, epochs=50, steps_per_epoch=6*500, save_freq=10, max_ep_len=6)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=1)  # number of models
    # parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--env_name', type=str, default="MNISTClassEnv-v0")
    parser.add_argument('--exp_name', type=str, default='ppo-1000e-012')
    parser.add_argument('--desired_outputs', type=list, default=[0, 1, 2])
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    run_experiment(args, ppo)


# RESULTS:
