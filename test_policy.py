from spinup.utils.test_policy import load_policy, run_policy
import envs
import gym


policy_name = 'ppo-idle-012'
_, get_action = load_policy('./data/{}/{}_s0'.format(policy_name, policy_name))
env = gym.make('MNISTClassEnv-v0')
run_policy(env, get_action)
