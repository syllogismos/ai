

import random
from tqdm import tqdm
import gym
import numpy as np
from gym import wrappers
from TileCoding import IHT, tiles

# env = gym.make('MountainCarContinuous-v0')
MAX_SIZE = 2048
NUM_OF_TILINGS = 8

alpha = 0.5/8
beta = 0.01


iht = IHT(MAX_SIZE)

def get_active_tiles(state, action, env):
    position = state[0]
    velocity = state[1]
    max_position, max_velocity = tuple(env.observation_space.high)
    min_position, min_velocity = tuple(env.observation_space.low)
    max_action = env.action_space.high[0]
    min_action = env.action_space.low[0]
    return tiles(iht, NUM_OF_TILINGS, [8*position/(max_position - min_position), \
                                        8*velocity/(max_velocity - min_velocity), \
                                        8*action[0]/(max_action - min_action)])
