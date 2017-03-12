"""
semi gradient sarsa control of Mountain Car
with function approximation, with tiled features

TileCoding -> https://gist.github.com/042bb46cc9143a0c027d021c552300cf

open ai gym evaluation: https://gym.openai.com/evaluations/eval_WOpAM1HyR621mN41t6MEpA#reproducibility
"""


import numpy as np
import gym
from gym import wrappers
from TileCoding import IHT, tiles
import random
from tqdm import tqdm
# from mountain_car import MountainCarEnv

import pickle

NUM_OF_TILINGS = 8
MAX_SIZE = 2014
MAX_EPISODES = 1200
EPSILON = 0.1
ALPHA = 0.01
GAMMA = 0.98
reg = 0.001

OUTDIR = '/tmp/MountainCar_v0-sarsa-results'

random.seed(0)
iht = IHT(MAX_SIZE)

# env = gym.make('MountainCar-v0')
weights = np.zeros(MAX_SIZE)

def get_value(state, action, weights, env):
    position = state[0]
    velocity = state[1]
    max_position, max_velocity = tuple(env.observation_space.high)
    min_position, min_velocity = tuple(env.observation_space.low)
    active_tiles = tiles(iht, NUM_OF_TILINGS, [8*position/(max_position - min_position), 8*velocity/(max_velocity - min_velocity)], [action])
    return sum(weights[active_tiles])  # + 0.5*reg*sum(weights * weights)

def get_active_tiles(state, action, env):
    position = state[0]
    velocity = state[1]
    max_position, max_velocity = tuple(env.observation_space.high)
    min_position, min_velocity = tuple(env.observation_space.low)
    return tiles(iht, NUM_OF_TILINGS, [8*position/(max_position - min_position), 8*velocity/(max_velocity - min_velocity)], [action])

def pick_epsilon_greedy_action(state, weights, env):
    actions = [0,1,2]
    if random.random() < EPSILON:
        return random.choice(actions)
    else:
        state_action_values = map(lambda a: get_value(state, a, weights, env), actions)
        max_value = max(state_action_values)
        return state_action_values.index(max_value)

def pick_best_action(state, weights, env):
    actions = [0,1,2]
    state_action_values = map(lambda a: get_value(state, a, weights, env), actions)
    max_value = max(state_action_values)
    return state_action_values.index(max_value)
    
def semi_gradient_sarsa():
    weights = np.zeros(MAX_SIZE)
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 50000
    # env = MountainCarEnv()
    # env = wrappers.Monitor(env, OUTDIR, force = True)
    for i in range(MAX_EPISODES):
        state = env.reset()
        action = pick_epsilon_greedy_action(state, weights, env)
        losses = []
        while True:
            next_state, reward, done, info = env.step(action)
            # print done, next_state, reward, action
            env.render()
            active_tiles = get_active_tiles(state, action, env)
            state_value = get_value(state, action, weights, env)
            if done:
                # print sum(weights), 'weights sum'
                # print done, info, next_state, reward
                # print "its done it seems", i
                delta = reward - state_value
                weights[active_tiles] += ALPHA * delta
                break
            next_action = pick_epsilon_greedy_action(next_state, weights, env)
            next_state_value = get_value(next_state, next_action, weights, env)
            delta = reward + (GAMMA * next_state_value) - state_value
            loss = delta**2
            losses.append(loss)
            weights[active_tiles] += ALPHA * delta
            # weights -= reg*weights
            state = next_state
            action = next_action
        print np.mean(losses), sum(weights)
        losses = []

    env.close()
    # gym.upload(OUTDIR)
    # pickle.dump(weights, open('MountainCar_v0_weights_120eps.pkl', 'wb'))
    return weights



if __name__ == '__main__':
    # Training sarsa weights
    weights = semi_gradient_sarsa()