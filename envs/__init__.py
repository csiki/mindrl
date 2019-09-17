import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='MNISTClassEnv-v0',
    entry_point='envs.mnist_class_env:MNISTClassEnv',
)
# TODO https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
