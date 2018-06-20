import tensorflow as tf
import numpy as np

class RL(object):
	def __init__(self):
		pass



class StagedDQN(RL):
	def __init__(self, state_space, action_spaces):
		super().__init__()

		self.state_space = state_space
		self.action_spaces = action_spaces
		self.stages = len(self.action_spaces)

		self.inputs, self.actions = self.build_network()

		# one hot indicator to control where loss is calculated
		self.indicator = tf.placeholder(tf.float32, [None, self.stages])

		self.r = tf.placeholder(tf.float32, [None])
		self.a = [tf.placeholder(tf.float32, [None, a_space]) for a_space in self.action_spaces]

		self.loss = tf.reduce_mean([self.indicator[:,i] * tf.square(self.r - tf.reduce_sum(self.a[i] * self.actions[i])) for i in range(self.stages)])


	def build_network(self):
		inputs = tf.placeholder(tf.float32, [None] + list(self.state_space))

		if len(inputs.shape) == 4:
			# Start with 3x3x9 input
			net = tf.layers.conv2d(inputs, 32, 1)
			net = tf.nn.relu(net)

			net = tf.layers.conv2d(net, 64, 3)
			net = tf.nn.relu(net)
			net = tf.squeeze(net, [1, 2])

		net = tf.layers.dense(net, 128)
		net = tf.nn.relu(net)

		actions = []

		for action_space in self.action_spaces:
			action = tf.layers.dense(net, action_space)

			actions.append(action)

		return inputs, actions

	def value(self, sess, inputs, stage):
		return sess.run(self.actions[stage], {self.inputs: inputs})

	# How to train multi stages at once? Sample both evenly from replay? Alternate training on a batch of one, then the other?
	# The reward for the 4th move in stage one should be the reward from the only move of the 2nd stage, which means there's a slight delay 
	# when we actually get back that reward
	def train(self, sess, inputs, stage, actions):
		pass



if __name__ == '__main__':

	state_space = (3, 3, 9)
	action_spaces = [9, 8]

	rl = StagedDQN(state_space, action_spaces)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		state = np.random.random(state_space)

		v0 = rl.value(sess, [state], 0)
		v1 = rl.value(sess, [state], 1)
		print(v0, v1, np.argmax(v0, 1), np.argmax(v1, 1))
