import argparse
from env import make
from model import StagedDQN
import numpy as np
from replay import StagedReplayBuffer

import tensorflow as tf

def train(args):
	env = make()
	network = StagedDQN(env.state_space, env.action_spaces)
	replay = StagedReplayBuffer(env.stages)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for episodeId in range(args.max_episodes):
			prev_state = env.reset()
			total_reward = 0
			total_history = []

			for stageId in range(env.stages):

				# keep around entire episode's history
				# At the end of the last stage, sum that stage's rewards, and 
				# Add that to the previous stage's reward for the last action
				# Back this up all the way to the beginning, and add all the training data
				# to the replay buffer

				stage_history = []

				# Each step
				while True:
					if np.random.random() < 0.5:
						action = np.argmax(network.value(sess, [prev_state], stageId)[0])
					else:
						action = np.random.choice(env.action_spaces[stageId])

					state, reward, done, info = env.step(action)

					total_reward += reward

					stage_history.append([prev_state, action, reward, done, state])

					prev_state = state

					if done:
						#print(episodeId, stageId, total_reward)
						break

				total_history.append(stage_history)

			#print(total_history)

			_, _, rewards, _, _ = list(zip(*total_history[-1]))

			for i in range(len(total_history) - 1)[::-1]:
				stage_reward = np.sum(rewards)
				_, _, rewards, _, _ = list(zip(*total_history[i]))
				#print('stage reward', stage_reward)

				total_history[i][-1][2] += stage_reward

			# Add history to replay
			for stageId, stage_history in enumerate(total_history):
				for stepId, step in enumerate(stage_history):
					s, a, r, t, s2 = step
					replay.add(stageId, s, a, r, t, s2)
					#print(j[2])
				#print()


			print(episodeId, total_reward)

			#for i in range(10):
				#print(replay.sample(0, 4))
				#print(replay.sample(1, 4))
				#pass

			if replay.size > args.batch_size:
				samples = [replay.sample(i, args.batch_size) for i in range(env.stages)]
				#print(len(samples))
				#return



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--max_episodes', default=10000, help='Maximum number of episodes to run.')
	parser.add_argument('--batch_size', default=32, help='Maximum number of episodes to run.')

	args = parser.parse_args()

	train(args)