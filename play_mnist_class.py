import gym
import envs
from spinup.utils.run_utils import ExperimentGrid
from spinup import ddpg, ppo, sac
import tensorflow as tf
import numpy as np


# TODO try other rl models

# from https://gist.github.com/krishpop/f4b2aa8d60d7b22bce8c258fd68ab11c
def run_experiment(args, rl_model):
    def env_fn():
        import envs  # registers custom envs to gym env registry
        return gym.make(args.env_name)

    eg = ExperimentGrid(name=args.exp_name)
    eg.add('env_fn', env_fn)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 80)
    # eg.add('steps_per_epoch', 50)  # FIXME
    eg.add('save_freq', 10)
    # eg.add('num_runs', args.num_runs)
    eg.add('max_ep_len', 6)  # FIXME n_steps // stim_stepsx
    eg.add('ac_kwargs:activation', tf.tanh, '')
    eg.run(rl_model, num_cpu=args.cpu, data_dir=args.data_dir)


if __name__ == '__main__':
    env = gym.make('MNISTClassEnv-v0')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=4)
    parser.add_argument('--env_name', type=str, default="MNISTClassEnv-v0")
    parser.add_argument('--exp_name', type=str, default='ppo-custom3-refactory-box')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    run_experiment(args, ppo)
