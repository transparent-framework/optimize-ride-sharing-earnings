import random
import numpy as np
from collections import deque


class TrajectoryBuffer(object):

    def __init__(self, buffer_size):
        self.max_size = buffer_size
        self.buffer = deque(maxlen=self.max_size)

    def push(self, env_trajectory):
        for trajectory in env_trajectory:
            self.buffer.append(trajectory)

    def sample(self, batch_size):
        if batch_size >= self.size():
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def size(self):
        return len(self.buffer)

    def sample_sequences(self, sequence_size, batch_size):
        batch = []

        max_start = 288 - sequence_size
        idx = np.random.choice(len(self.buffer), size=batch_size)
        starts = np.random.choice(range(max_start), size=batch_size)
        for i in range(batch_size):
            traj = self.buffer[idx[i]]
            start = starts[i]
            batch.append(traj[start:start+sequence_size])

        return batch
