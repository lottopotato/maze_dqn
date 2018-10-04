"""
DQN Class
reference by https://github.com/hunkim/ReinforcementZeroToAll/ 
It is also reference by here. 
DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""

import numpy as np
import tensorflow as tf

class DQN:
	def __init__(self, session:tf.Session, input_array:np.ndarray, output_size:int,
	 name:str = 'main'):
		self.session = session
		self.output_size = output_size
		self.net_name = name
		self.data_x = input_array[0]
		self.data_y = input_array[1]
		self.network()

	def network(self, learning_rate = 1e-3):
		with tf.variable_scope(self.net_name):
			self.X = tf.placeholder(tf.float32, [None, self.data_x, self.data_y, 1])
			opservation = self.X
			self.Y = tf.placeholder(tf.float32, [None, self.output_size])
			# 17 17
			conv1 = tf.layers.conv2d(
				inputs = opservation, filters = 32, kernel_size = [2,2],
				padding = "valid", activation = tf.nn.relu)
			# 16 16 
			pool1 = tf.layers.max_pooling2d(
				inputs = conv1, pool_size = [2,2], strides = 2)
			# 8 8
			conv2 = tf.layers.conv2d(
				inputs = pool1, filters = 64, kernel_size = [2,2],
				padding = "same", activation = tf.nn.relu)
			pool2 = tf.layers.max_pooling2d(
				inputs = conv2, pool_size = [2,2], strides = 2)
			# 4 4
			conv3 = tf.layers.conv2d(
				inputs = pool2, filters = 128, kernel_size = [2,2],
				padding = "same", activation = tf.nn.relu)
			pool3 = tf.layers.max_pooling2d(
				inputs = conv3, pool_size = [2,2], strides = 2)
			# 2 2 
			pool3_flat3 = tf.reshape(pool3, [-1, 2*2*128])
			fc4 = tf.layers.dense(
				inputs = pool3_flat3, units = 256, activation = tf.nn.relu)
			fc5 = tf.layers.dense(
				inputs = fc4, units = self.output_size)

			self.Qpred = fc5
			self.loss = tf.losses.mean_squared_error(self.Y, self.Qpred)
			train = tf.train.AdamOptimizer(learning_rate)
			self.train_op = train.minimize(self.loss)

	def predict(self, state):
		state = np.reshape(state, [-1, self.data_x, self.data_y, 1])
		return self.session.run(self.Qpred, feed_dict = {self.X:state})

	def update(self, x_stack, y_stack):
		x_stack = np.reshape(x_stack, [-1, self.data_x, self.data_y, 1])
		return self.session.run([self.loss, self.train_op],
			feed_dict = {self.X:x_stack, self.Y:y_stack})





