from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

import mario_wrappers
import atari_wrappers
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#env = mario_wrappers.wrapper(env)
env = atari_wrappers.wrap_deepmind(env, episode_life= False)
done = True


for step in range(500000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    if done==True:
        import pdb
        pdb.set_trace()
    env.render()

env.close()