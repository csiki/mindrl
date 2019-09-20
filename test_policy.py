from spinup.utils.test_policy import load_policy, run_policy
import envs
import gym


_, get_action = load_policy('./data/ddpg-custom3-50epoch/ddpg-custom3-50epoch_s0')
env = gym.make('MNISTClassEnv-v0')
run_policy(env, get_action)

