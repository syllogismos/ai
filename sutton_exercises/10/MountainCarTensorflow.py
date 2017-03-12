"""
SARSA with tensorflow on mountain car
"""

import numpy as np
import gym
from gym import wrappers
from TileCoding import IHT, tiles
import random
from tqdm import tqdm
import tensorflow as tf

import pickle

NUM_OF_TILINGS = 8
MAX_SIZE = 1024
MAX_EPISODES = 1200
EPSILON = 0.1
ALPHA = 0.01
GAMMA = 0.98
OUT_DIR = '/tmp/mountaincar_tf_no_biases/'

random.seed(0)


class TensorFlowValueEstimatorPartialGradient(object):
    def __init__(self, env):
        self.env = env
        self.iht = IHT(MAX_SIZE)
        self.weights = tf.Variable(tf.zeros((1, MAX_SIZE)))
        # self.weights = tf.get_variable('weights', [1, MAX_SIZE])
        # self.biases = tf.Variable(tf.zeros((1,1)))
        self.tile_features = tf.placeholder(tf.float32, name='tile_features')
        self.target = tf.placeholder(tf.float32, name='target')
        self.value_approx = tf.reduce_sum(tf.matmul(self.weights, self.tile_features)[0][0])# + self.biases)
        self.subtract = self.value_approx - self.target
        self.loss = self.subtract**2
        self.log_loss = tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=ALPHA)
        self.gradient = self.optimizer.compute_gradients(self.value_approx)
        self.partial_gradient = map(lambda x: (self.subtract * x[0], x[1]), self.gradient)
        self.train_op = self.optimizer.apply_gradients(self.partial_gradient)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter('mountaincar_tf/' + str(ALPHA), self.sess.graph)

        self.old_velocity = 0.0
    
    def get_wts(self):
        x = self.sess.run(self.weights)
        return np.sum(x**2.0)

    def predict(self, state, action):
        # sess = tf.get_default_session()
        tile_features = self.get_active_tiles_tensor(state, action)
        return self.sess.run(self.value_approx, {self.tile_features: tile_features})

    def update(self, state, action, target):
        # sess = tf.get_default_session()
        tiles_features = self.get_active_tiles_tensor(state, action)
        self.old_velocity = state[1]
        _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary_op], {
            self.tile_features: tiles_features,
            self.target: target})
        self.writer.add_summary(summary)
        return loss
    
    def get_active_tiles_tensor(self, state, action):
        position = state[0]
        velocity = state[1]
        acceleration = state[1] - self.old_velocity
        max_position, max_velocity = tuple(self.env.observation_space.high)
        min_position, min_velocity = tuple(self.env.observation_space.low)
        max_acceleration = max_velocity - min_velocity
        continuous_features = [8*position/(max_position - min_position),
                               8*velocity/(max_velocity - min_velocity),
                               8*acceleration/(2*max_acceleration)]
        tile_nos = tiles(self.iht, NUM_OF_TILINGS, continuous_features, [action])
        tile_array = np.zeros((MAX_SIZE, 1))
        tile_array[tile_nos] = 1.0
        return tile_array

    def get_epsilon_greedy_action(self, state):
        actions = range(self.env.action_space.n)
        if random.random() < EPSILON:
            action = random.choice(actions)
            q_value = self.predict(state, action)
            return action, q_value
        q_values = map(lambda x: self.predict(state, x), actions)
        max_q_value = max(q_values)
        return q_values.index(max_q_value), max_q_value







class TensorFlowValueEstimatorWrong(object):







    def __init__(self, env):
        self.env = env
        self.iht = IHT(MAX_SIZE)
        self.weights = tf.Variable(tf.zeros((1, MAX_SIZE)))
        # self.weights = tf.get_variable('weights', [1, MAX_SIZE])
        # self.biases = tf.Variable(tf.zeros((1,1)))
        self.tile_features = tf.placeholder(tf.float32, name='tile_features')
        self.target = tf.placeholder(tf.float32, name='target')
        self.action_value = tf.reduce_sum(tf.matmul(self.weights, self.tile_features)[0][0])# + self.biases)
        self.subtract = self.target - self.action_value
        self.update_op = tf.add(self.weights[self.tile_features], ALPHA*self.subtract)
        self.assign = tf.assign(self.weights, self.update_op)
        self.loss = tf.sqrt(tf.square(tf.subtract(self.action_value, self.target)))
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = ALPHA)
        # self.grads = self.optimizer.compute_gradients(self.loss)
        # self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
        # tf.get_default_session().run(tf.global_variables_initializer())
        # self.delta = tf.placeholder(tf.float32)
        # self.coeff = ALPHA*self.subtract
        # self.update_op = tf.add(-self.grads[0][0]*self.coeff, self.weights)
        # self.assign = tf.assign(self.weights, self.update_op)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.old_velocity = 0.0
    
    def get_wts(self):
        x = self.sess.run(self.weights)
        return np.sum(x**2.0)

    def predict(self, state, action):
        # sess = tf.get_default_session()
        tile_features = self.get_active_tiles_tensor(state, action)
        return self.sess.run(self.action_value, {self.tile_features: tile_features})

    def update(self, state, action, target, value):
        # sess = tf.get_default_session()
        tiles_features = self.get_active_tiles_tensor(state, action)
        self.old_velocity = state[1]
        _, loss = self.sess.run([self.assign, self.loss], {
            self.tile_features: tiles_features,
            self.target: target})
        
        return loss
    
    def get_active_tiles_tensor(self, state, action):
        position = state[0]
        velocity = state[1]
        acceleration = state[1] - self.old_velocity
        max_position, max_velocity = tuple(self.env.observation_space.high)
        min_position, min_velocity = tuple(self.env.observation_space.low)
        max_acceleration = max_velocity - min_velocity
        continuous_features = [8*position/(max_position - min_position),
                               8*velocity/(max_velocity - min_velocity),
                               8*acceleration/(2*max_acceleration)]
        tile_nos = tiles(self.iht, NUM_OF_TILINGS, continuous_features, [action])
        tile_array = np.zeros((MAX_SIZE, 1))
        tile_array[tile_nos] = 1.0
        # sparse_tensor = tf.SparseTensor(indices = map(lambda x: [0, x], tile_nos),
        #                                     values = [1.0]*len(tile_nos),
        #                                     shape = [1,MAX_SIZE])
        return tile_array

    def get_epsilon_greedy_action(self, state):
        actions = range(self.env.action_space.n)
        if random.random() < EPSILON:
            action = random.choice(actions)
            q_value = self.predict(state, action)
            return action, q_value
        q_values = map(lambda x: self.predict(state, x), actions)
        max_q_value = max(q_values)
        return q_values.index(max_q_value), max_q_value

class NormalValueEstimator(object):
    def __init__(self, env):
        self.env = env
        self.weights = np.zeros(MAX_SIZE)
        self.iht = IHT(MAX_SIZE)
    
    def get_active_tiles(self, state, action):
        position = state[0]
        velocity = state[1]
        max_position, max_velocity = tuple(self.env.observation_space.high)
        min_position, min_velocity = tuple(self.env.observation_space.low)
        continuous_features = [8*position/(max_position - min_position),
                               8*velocity/(max_velocity - min_velocity)]
        return tiles(self.iht, NUM_OF_TILINGS, continuous_features, [action])
    

    def predict(self, state, action):
        tiles_nos = self.get_active_tiles(state, action)
        return sum(self.weights[tiles_nos])
        
    def update(self, state, action, target, value):
        tiles_nos = self.get_active_tiles(state, action)
        # print tiles_nos
        self.weights[tiles_nos] += ALPHA * (target-value)
        # print target
        # print value
        # print target-value
        return np.sqrt((target-value)**2)

    def get_epsilon_greedy_action(self, state):
        actions = range(self.env.action_space.n)
        if random.random() < EPSILON:
            action = random.choice(actions)
            q_value = self.predict(state, action)
        q_values = map(lambda x: self.predict(state, x), actions)
        max_q_value = max(q_values)
        return q_values.index(max_q_value), max_q_value










def partial_gradient_sarsa(env, upload=False):
    env = gym.make('MountainCar-v0')
    if upload:
        env = wrappers.Monitor(env, OUT_DIR, force=True)
    value_estimator = TensorFlowValueEstimatorPartialGradient(env)
    for i in tqdm(range(MAX_EPISODES)):
        # print 'weights', sum(value_estimator.weights.eval()[0])
        print 'weights', value_estimator.get_wts()
        # print 'weights', sum(value_estimator.weights)
        state = env.reset()
        action, q_value = value_estimator.get_epsilon_greedy_action(state)
        losses = []
        while True:
            # print action, loss
            next_state, reward, done, info = env.step(action)
            env.render()
            if done:
                target = reward
                loss = value_estimator.update(state, action, target)
                losses.append(loss)
                # print losses, 'losses'
                if i % 50 == 0:
                    print sum(losses)/len(losses)
                losses = []
                break
            next_action, next_q_value = value_estimator.get_epsilon_greedy_action(next_state)
            target = reward + (GAMMA * next_q_value)
            loss = value_estimator.update(state, action, target)
            losses.append(loss)
            state = next_state
            action = next_action
    env.close()
    value_estimator.writer.close()
    if upload:
        gym.upload(OUT_DIR)



if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.spec.max_episode_steps = 500
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        partial_gradient_sarsa(env, upload=False)