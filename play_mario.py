import gym
from mario_wrappers_v2 import wrap_environment
from mario_wrappers import make_env
import random
import storage
import torch
import DQN_agent
import Double_DQN_agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY


"""
Where we actually run the whole process of learning from the environment

We'll have 
1. the agent perform real actions within the environment
2. store the performed SARSA in the replay buffer
3. At every update step, perform optimization
4. 
eps_start=1.0
eps_decay=.999985
eps_min=0.02
"""

if __name__=='__main__':


    # Call the Environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    #env = atari_wrappers.wrap_deepmind(env)
    #env = JoypadSpace(env, SIMPLE_MOVEMENT)
    #env = JoypadSpace(env, RIGHT_ONLY)
    env = wrap_environment(env)
    #env = make_env(env)
    # Thanks to the wrapper, we now get 84*84*4 observation each step

    log_name = 'DQN-Mario_wrapper_v2_normalize10-flag_350-dead_50-xpos-clip_grad'


    # Get cuda device
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Call the storage
    observation = env.reset()
    storage = storage.Storage(np.array(observation).shape, num_steps = 30000, device = device)

    # Call Tensorboard logger
    writer = SummaryWriter('runs/' + log_name +time.strftime("%d-%m-%Y_%H-%M-%S"))

    # Call the Agent
    #agent = DQN_agent.Agent(env,log_name, storage, explore = 0.02, lmbda=0.99, num_actions = env.action_space.n, device = device, explore_timesteps = 200000, writer = writer)
    agent = Double_DQN_agent.Agent(env, log_name, storage, explore = 0.02, lmbda=0.9, num_actions = env.action_space.n, device = device, explore_timesteps = 200000, writer = writer)

    # Train the Agent
    agent.evaluate()

    # I guess this will be it?



