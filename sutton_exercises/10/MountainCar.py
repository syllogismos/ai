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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# from mountain_car import MountainCarEnv

import pickle

NUM_OF_TILINGS = 8
MAX_SIZE = 2014
MAX_EPISODES = 20000
EPSILON = 0.1
ALPHA = 0.01
GAMMA = 0.98
reg = 0.001

OUTDIR = '/tmp/MountainCar_v0-sarsa-results'

random.seed(0)
iht = IHT(MAX_SIZE)

# env = gym.make('MountainCar-v0')
weights = np.zeros(MAX_SIZE)

def get_epsilon(iteration):
    if iteration < 100:
        return 0.2
    elif iteration < 5000:
        return 0.1
    elif iteration < 10000:
        return 0.01
    else: return 0.001

def get_q_value(state, action, weights, env):
    active_tiles = get_active_tiles(state, action, env)
    # active_tiles = tiles(iht, NUM_OF_TILINGS, [8*position/(max_position - min_position), 8*velocity/(max_velocity - min_velocity)], [action])
    return sum(weights[active_tiles]), active_tiles  # + 0.5*reg*sum(weights * weights)

def get_active_tiles(state, action, env):
    position = state[0]
    velocity = state[1]
    x3 = state[2]
    x4 = state[3]
    max_position, max_velocity, max_x3, max_x4 = tuple(env.observation_space.high)
    min_position, min_velocity, min_x3, min_x4 = tuple(env.observation_space.low)
    return tiles(iht, NUM_OF_TILINGS, [8*position/(max_position - min_position),
                                       8*velocity/(max_velocity - min_velocity),
                                       8*x3/(max_x3 - min_x3),
                                       8*x4/(max_x4 - min_x4)], [action])

def pick_epsilon_greedy_action(state, weights, env, i):
    actions = [0,1]
    if random.random() < get_epsilon(i):
        return random.choice(actions)
    else:
        state_action_values = map(lambda a: get_q_value(state, a, weights, env)[0], actions)
        max_value = max(state_action_values)
        return state_action_values.index(max_value)

def pick_best_action(state, weights, env):
    actions = [0,1]
    state_action_values = map(lambda a: get_q_value(state, a, weights, env)[0], actions)
    max_value = max(state_action_values)
    return state_action_values.index(max_value)
    
def semi_gradient_sarsa(env, upload=False):
    weights = np.zeros(MAX_SIZE)
    # env = gym.make('MountainCar-v0')
    # env._max_episode_steps = 50000
    if upload:
        env = wrappers.Monitor(env, OUTDIR, force = True)
    for i in tqdm(range(MAX_EPISODES)):
        state = env.reset()
        action = pick_epsilon_greedy_action(state, weights, env, i)
        losses = []
        episode_length = 0
        while True:
            episode_length += 1
            next_state, reward, done, info = env.step(action)
            env.render()
            state_value, active_tiles = get_q_value(state, action, weights, env)
            if done:
                delta = reward - state_value
                weights[active_tiles] += ALPHA * delta
                break
            next_action = pick_epsilon_greedy_action(next_state, weights, env, i)
            next_state_value = get_q_value(next_state, next_action, weights, env)[0]
            delta = reward + (GAMMA * next_state_value) - state_value
            loss = delta**2
            losses.append(loss)
            weights[active_tiles] += ALPHA * delta
            # weights -= reg*weights
            state = next_state
            action = next_action
        # print np.mean(losses), sum(weights)
        if (i+1)%100 == 0:
            print episode_length
        losses = []

    env.close()
    if upload:
        gym.upload(OUTDIR)
    # pickle.dump(weights, open('MountainCar_v0_weights_120eps.pkl', 'wb'))
    return weights

def plot_cost_to_go_mountain_car(env, weights, num_tiles=20):
    x = np.random.uniform(env.observation_space.low[0], env.observation_space.high[0], num_tiles)
    y = np.random.uniform(env.observation_space.low[1], env.observation_space.high[1], num_tiles)
    X, Y = np.meshgrid(x, y)
    # Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    Z = np.zeros(len(x))
    for i in range(len(x)):
        Z[i] = pick_best_action([x[i], y[i]], weights, env)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()

if __name__ == '__main__':
    # Training sarsa weights
    env = gym.make('CartPole-v0')
    weights = semi_gradient_sarsa(env, True)
    # plot_cost_to_go_mountain_car(env, weights, 250)