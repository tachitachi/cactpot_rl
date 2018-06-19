import numpy as np


# Two stage game
# 
# 1. Pick 4 spaces
# 2. Pick row/column/angle


class Cactpot(object):

	point_map = {
		6: 10000,
		7: 36,
		8: 720,
		9: 360,
		10: 80,
		11: 252,
		12: 108,
		13: 72,
		14: 54,
		15: 180,
		16: 72,
		17: 180,
		18: 119,
		19: 36,
		20: 306,
		21: 1080,
		22: 144,
		23: 1800,
		24: 3600,
	}

	width = height = 3
	size = width * height

	def __init__(self):
		self.board = np.arange(Cactpot.size) + 1
		self.mask = np.zeros(Cactpot.size)

		self.turn = 0

		self.state_space = (Cactpot.height, Cactpot.height, Cactpot.size)
		self.action_space = Cactpot.size

		# r1, r2, r3, c1, c2, c3, xlr, xrl
		self.action_space_2 = Cactpot.height + Cactpot.width + 2

	def _state(self):
		z = np.zeros((Cactpot.size, Cactpot.size))
		z[np.arange(Cactpot.size), self.board - 1] = 1
		z[np.where(~self.mask.astype(np.bool))] = 0
		return z.reshape((Cactpot.height, Cactpot.width, Cactpot.size))

	def step(self, action):

		done = False
		reward = 0

		if self.turn < 4:
			assert(action < Cactpot.size)
			if self.mask[action] == 1:
				reward = -100
			self.mask[action] = 1
			self.turn = np.sum(self.mask).astype(np.int32)

			if self.turn == 4:
				done = True

		elif self.turn == 4:
			assert(action < Cactpot.height + Cactpot.width + 2)
			m = np.zeros((Cactpot.height, Cactpot.height))
			board = np.reshape(self.board, (Cactpot.height, Cactpot.height))
			if action < Cactpot.height:
				s = np.sum(board[action,:])
			elif action < Cactpot.height + Cactpot.width:
				action = action - Cactpot.height
				s = np.sum(board[:,action])
			elif action == Cactpot.height + Cactpot.width:
				s = np.sum(board[np.arange(Cactpot.height), np.arange(Cactpot.width)])
			elif action == Cactpot.height + Cactpot.width + 1:
				s = np.sum(board[np.arange(Cactpot.height), np.arange(Cactpot.width)[::-1]])
			else:
				raise IndexError('Invalid action')

			self.mask[:] = 1

			reward = Cactpot.point_map[s]
			done = True


		state = self._state()
		info = {}

		return state, reward, done, info

	def render(self):
		b = (self.board * self.mask).reshape((Cactpot.height, Cactpot.width)).astype(np.int32)
		for row in b:
			print(row.tolist())
		print()

	def reset(self):
		np.random.shuffle(self.board)
		self.mask = np.zeros(Cactpot.size)

		self.turn = 0

		return self._state()

def make():
	return Cactpot()

if __name__ == '__main__':
	env = make()

	state = env.reset()

	env.render()

	env.step(0)
	env.step(1)
	env.step(3)
	state, reward, done, info = env.step(4)

	print(state, reward)

	env.render()

	state, reward, done, info = env.step(5)

	print(state, reward)