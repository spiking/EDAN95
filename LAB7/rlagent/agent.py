"""agent.py: Contains the entire deep reinforcement learning agent."""
__author__ = "Erik Gärtner"

from collections import deque

import tensorflow as tf
import numpy as np
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from .expreplay import ExpReplay


class Agent():
    """
    The agent class where you should implement the vanilla policy gradient agent.
    """
    # LunarLander: state_size = 8, action_size = 4
    def __init__(self, tf_session, state_size=(4,), action_size=2,
                 learning_rate=1e-3, gamma=0.99, memory_size=5000):
        """
        The initialization function. Besides saving attributes we also need
        to create the policy network in Tensorflow that later will be used.
        """

        self.state_size = state_size[0]
        self.action_size = action_size
        self.tf_sess = tf_session
        self.gamma = gamma
        self.replay = ExpReplay(memory_size)

        with tf.variable_scope('agent'):

            # Create tf placeholders, i.e. inputs into the network graph.
            init = tf.contrib.layers.variance_scaling_initializer()

            self.state = tf.placeholder(tf.float32, shape=[None, self.state_size], name='state')
            self.actions = tf.placeholder(tf.int32, shape=[None, self.action_size], name="actions")
            self.discounted_episode_rewards = tf.placeholder(tf.float32, shape=[None], name="discounted_episode_rewards")

            # Layers
            self.input_layer = tf.layers.dense(self.state, 128, activation=tf.nn.relu, kernel_initializer=None)
            self.hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=None)
            self.outputs = tf.layers.dense(self.input_layer, self.action_size, activation=tf.nn.relu, kernel_initializer=None)
            self.outputs_softmax = tf.nn.softmax(self.outputs)

            # Negative log probability of our actions as an output, multiplied by discounted reward
            self.log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=self.actions)
            self.loss = tf.reduce_mean(self.log_prob * self.discounted_episode_rewards)

            # Minimize loss
            self.adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.adam.minimize(self.loss)


        # Init
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.tf_sess.run(self.init)

    def take_action(self, state):
        """
        Given the current state sample an action from the policy network.
        Return a the index of the action [0..N).
        """
        # Forward prop to get softmax action probabilities
        action_distribution = self.tf_sess.run([self.outputs_softmax], feed_dict={self.state: state.reshape(1, self.state_size)})[0][0]
        # Sample action from distribution
        action = np.random.choice(range(self.action_size), p=action_distribution)

        return action

    def convert_action(self, action):
        a = np.zeros(self.action_size)
        a[action] = 1
        return a.reshape(1, self.action_size)

    def record_action(self, state0, action, reward, state1, done):
        """
        Record an action taken by the action and the associated reward
        and next state. This will later be used for traning.
        """
        action = self.convert_action(action)

        self.replay.add([state0, action, reward, state1, done])

    def discounted_reward_and_normalized(self, rewards):
        """
        Given the rewards for an epsiode discount them by gamma.
        Next since we are sending them into the neural network they should
        have a zero mean and unit variance.

        Return the new list of discounted and normalized rewards.
        """

        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0

        for step in reversed(range(len(rewards))):
            cumulative_rewards = cumulative_rewards + self.gamma * rewards[step]
            discounted_rewards[step] = cumulative_rewards

        discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
        discounted_rewards = discounted_rewards / np.std(discounted_rewards)

        return discounted_rewards

    def train_agent(self):

        """
        Train the policy network using the collected experiences during the
        episode(s).
        """

        # Retrieve collected experiences from memory
        memory = self.replay.get_all()

        # Extract rewards
        rewards = [exp[2] for exp in memory]
        normalized_rewards = self.discounted_reward_and_normalized(rewards)

        # Feed the experiences through the network with rewards to compute and
        # minimize the loss.

        states = np.array([exp[0] for exp in memory])
        actions = np.array([exp[1] for exp in memory])
        rewards = np.array([r for r in normalized_rewards])

        actions = actions.reshape(-1, self.action_size)

        [self.tf_sess.run(self.train_op, feed_dict={
                                self.state: states,
                                self.actions: actions,
                                self.discounted_episode_rewards: rewards}) for i in range(3)]

        self.replay.clear()
