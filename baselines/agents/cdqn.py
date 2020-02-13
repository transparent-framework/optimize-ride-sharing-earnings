import numpy as np
import torch
import torch.nn as nn
from buffers.cdqn import ReplayBuffer
from networks.cdqn import DQN


class cDQNAgent(object):

    def __init__(self, env, learning_rate, gamma, buffer_size, input_dim, output_dim):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)

        self.hex_attr_df = self.env.hex_attr_df[['hex_id', 'north_east_neighbor', 'north_neighbor',
                                                 'north_west_neighbor', 'south_east_neighbor',
                                                 'south_neighbor', 'south_west_neighbor']]
        self.neighbor_matrix = self.compute_neighbor_matrix()

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.model = DQN(input_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.MSE_loss = nn.MSELoss()

    def compute_neighbor_matrix(self):
        neighbor_matrix = np.ndarray((251, 6))  # 251 hex bins and 6 neighbors per bin
        for hex_bin in range(len(self.hex_attr_df)):
            neighbor_matrix[hex_bin] = self.hex_attr_df.iloc[hex_bin].values[1:]
        return neighbor_matrix

    def get_valid_action(self, valid_neighbors, qvals):
        # Applies valid action for hex bins on the map boundary
        # Only gives relocate action to neighboring bin if the
        # q-value of the neighboring bin is higher than current bin

        for i in range(0, 6):
            if (valid_neighbors[i] == 0) or (qvals[i] < qvals[6]):
                qvals[i] = -np.inf
        return np.argmax(qvals)

    def get_action(self, hex_bin, state, eps=0.10):
        neighbors = self.neighbor_matrix[hex_bin]
        valid_neighbors = np.isfinite(neighbors).astype(int)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)

        action = self.get_valid_action(valid_neighbors, qvals.cpu().detach().numpy()[0])

        if (np.random.randn() < eps):
            return np.random.choice(a=np.nonzero(valid_neighbors)[0], size=1, replace=False)[0]

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
