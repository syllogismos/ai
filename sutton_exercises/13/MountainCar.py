"""
Policy Gradient Methods
"""

import numpy as np
import gym
import tensorflow as tf
from TileCoding import IHT, tiles
import random
from tqdm import tqdm

import pickle

NUM_OF_TILINGS = 8
MAX_SIZE = 1024
MAX_EPISODES = 1200
EPSILON = 0.1
GAMMA = 0.98
ALPHA = 0.001




class GeneralPolicyEstimator(object):
    def __init__(self, env):
        self.env = env
        self.state_dim = MAX_SIZE
        self.iht = IHT(MAX_SIZE)
        self.num_actions = self.env.action_space.n
        self.gamma_coefficient = map(lambda x: GAMMA**x, range(100))
        self.states = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        self.actions = tf.placeholder(tf.float32, shape=[None, self.num_actions])
        self.advantages = tf.placeholder(tf.float32, shape=[None, 1])

        self.W1 = tf.get_variable('W1', [self.state_dim, 20],
                        initializer=tf.random_normal_initializer())
        self.b1 = tf.get_variable('b1', [20],
                        initializer=tf.constant_initializer(0))
        self.h1 = tf.nn.relu(tf.matmul(self.states, self.W1) + self.b1)
        self.W2 = tf.get_variable('W2', [20, self.num_actions],
                        initializer=tf.random_normal_initializer())
        self.b2 = tf.get_variable('b2', [self.num_actions],
                        initializer=tf.constant_initializer())
        self.probs = tf.nn.softmax(tf.matmul(self.h1, self.W2) + self.b2)

        self.loss = -tf.reduce_mean(tf.reduce_sum(tf.log(self.probs)*self.actions, 1, keep_dims=True)*self.advantages)
        # self.loss = -tf.reduce_mean(tf.reduce_sum(tf.log(self.probs), 1, keep_dims=True)*self.advantages)
        # self.loss = -tf.reduce_mean(tf.reduce_sum(tf.log(self.probs*self.actions), 1, keep_dims=True)*self.advantages)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA)

        self.train_op = self.optimizer.minimize(self.loss)
        self.sess = tf.Session()
        self.value_network()
        self.sess.run(tf.global_variables_initializer())
        tf.get_variable_scope().reuse_variables()
    
    def value_network(self):
        pass
    def update_value_estimator(self, state_features, rewards):
        pass
    def value_estimator(self, states):
        return np.zeros((len(states), 1))

    def predict(self, state):
        # sess = tf.get_default_session()
        if random.random() < EPSILON:
            return random.choice(range(self.num_actions))
        state_tiles = self.get_features([state])
        tf.get_variable_scope().reuse_variables()
        probs = self.sess.run(self.probs, {self.states: state_tiles})
        # print probs
        return probs.argmax()
        
    def generateEpisode(self, state):
        episode = []
        while True:
            self.env.render()
            action = self.predict(state)
            next_state, reward, done, info = self.env.step(action)
            # reward += next_state[0]
            if done:
                break
            episode.append((state, action, reward))
            state = next_state
        state_features = np.vstack(map(lambda x: self.get_features([x[0]]), episode))
        actions = np.vstack(map(lambda x: self.one_hot_action(x[1]), episode))
        advantages = []
        rewards = map(lambda x: x[2], episode)
        for i, reward in enumerate(rewards):
            advantages.append(sum(map(lambda x: x[0]*x[1], zip(rewards[i:], self.gamma_coefficient))))
        advantages = np.array(advantages)
        advantages = advantages.reshape(-1, 1)
        state_values = self.value_estimator(map(lambda x: x[0], episode))
        advantages -= state_values
        self.update_value_estimator(state_features, rewards)
        return episode, state_features, actions, advantages
    
    def get_features(self, states):
        # print states
        return np.vstack(map(lambda x: self.get_tiles(x), states))

    def get_tiles(self, state):
        position = state[0]
        velocity = state[1]
        max_position, max_velocity = tuple(self.env.observation_space.high)
        min_position, min_velocity = tuple(self.env.observation_space.low)
        continuous_features = [8*position/(max_position - min_position),
                               8*velocity/(max_velocity - min_velocity)]
        tile_nos = tiles(self.iht, NUM_OF_TILINGS, continuous_features)
        tile_array = np.zeros((1, MAX_SIZE))
        tile_array[:,tile_nos] = 1.0
        return tile_array              
    
    def update_policy(self, states, actions, advantages):
        _, loss = self.sess.run([self.train_op, self.loss], {self.states: states,
                                 self.actions: actions,
                                 self.advantages: advantages})
        return loss
        
    def one_hot_action(self, action):
        act = np.zeros(self.num_actions)
        act[action] = 1.0
        return act

class baselinePolicyEstimator(GeneralPolicyEstimator):

    def value_network(self):
        self.value_W1 = tf.get_variable('value_W1', [self.state_dim, 10],
                            initializer=tf.random_normal_initializer())
        self.value_b1 = tf.get_variable('value_b1', [10],
                            initializer=tf.constant_initializer())
        self.value_h1 = tf.nn.relu(tf.matmul(self.states, self.value_W1) + self.value_b1)
        self.value_W2 = tf.get_variable('value_W2', [10,1],
                            initializer=tf.random_normal_initializer())
        self.value_b2 = tf.get_variable('value_b2', [1], initializer=tf.constant_initializer())
        self.value = tf.matmul(self.value_h1, self.value_W2) + self.value_b2

        self.target_values = tf.placeholder(tf.float32, [None, 1])
        self.diffs = self.value - self.target_values
        self.value_loss = tf.nn.l2_loss(self.diffs)
        self.value_optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.value_loss)
        # tf.get_variable_scope().reuse_variables()
        # self.sess.run(tf.global_variables_initializer())
        pass

    def value_estimator(self, states):
        tf.get_variable_scope().reuse_variables()
        state_tiles = self.get_features(states)
        state_values = self.sess.run(self.value, {self.states: state_tiles})
        return state_values
        pass

    def update_value_estimator(self, states_features, rewards):
        state_values = self.sess.run(self.value, {self.states: states_features})
        target_values = np.vstack((state_values[1:, ], [[0.0]]))
        target_values = GAMMA * target_values + np.array(rewards).reshape(-1, 1)
        _, loss = self.sess.run([self.value_optimizer, self.value_loss], {
            self.states: states_features, self.target_values: target_values
        })
        print loss, 'value_loss'







if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 500
    pg = baselinePolicyEstimator(env)
    ss, ass, ads = [], [], []
    for i in tqdm(range(10000)):
        state = env.reset()
        e, s, a, ad = pg.generateEpisode(state)
        ss.append(s)
        ass.append(a)
        ads.append(ad)
        if i != 0 and i % 20 == 0:
            loss = pg.update_policy(np.vstack(ss), np.vstack(ass), np.vstack(ads))
            ss, ass, ads = [], [], []
            # if i % 10 == 0:
            print i, 'episode', loss, 'loss'







    