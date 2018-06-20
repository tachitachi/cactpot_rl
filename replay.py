import numpy as np

class ReplayBuffer(object):
	def __init__(self, max_size=50000):
		self.buffer = []
		self.max_size = max_size

	@property
	def size(self):
		return len(self.buffer)

	def add(self, s, a, r, t, s2):
		self.buffer.append((s, a, r, t, s2))

		if len(self.buffer) > self.max_size:
			self.buffer.pop(0)

	def sample(self, batch_size):
		assert(self.size >= batch_size)
		indices = np.random.choice(self.size, batch_size)

		items = np.take(np.array(self.buffer, dtype=object), indices, axis=0)

		s, a, r, t, s2 = list(zip(*items))

		return s, a, r, t, s2


class StagedReplayBuffer(object):
	def __init__(self, stage_count, max_size=50000):
		self.stage_count = stage_count
		self.buffers = []

		for i in range(self.stage_count):
			self.buffers.append(ReplayBuffer(max_size))

	def add(self, stage, s, a, r, t, s2):
		self.buffers[stage].add(s, a, r, t, s2)

	def sample(self, stage, batch_size):
		return self.buffers[stage].sample(batch_size)

	@property
	def size(self):
		return np.min([buf.size for buf in self.buffers])


if __name__ == '__main__':
	buf = ReplayBuffer()

	state_size = (3, 3, 9)

	for i in range(100):
		s = np.random.random(state_size)
		a = np.random.choice(9)
		r = np.random.random() * 100
		t = np.round(np.random.random()).astype(np.bool)
		s2 = np.random.random(state_size)

		buf.add(s, a, r, t, s2)

	s_, a_, r_, t_, s2_ = buf.sample(4)

	print(s_, a_, r_, t_, s2_)