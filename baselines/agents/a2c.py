import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from buffers.a2c import TrajectoryBuffer
from networks.ca2c import ValueNetwork  # Same as cA2C networks
from networks.ca2c import PolicyNetwork


class A2CAgent(object):

    def __init__(self, env, learning_rate, gamma, tau, buffer_size, input_dim, output_dim):
        self.logger = logging.getLogger("baseline_logger")
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        self.trajectory_buffer = TrajectoryBuffer(buffer_size)
        self.hex_attr_df = self.env.hex_attr_df[['hex_id', 'north_east_neighbor', 'north_neighbor',
                                                 'north_west_neighbor', 'south_east_neighbor',
                                                 'south_neighbor', 'south_west_neighbor']]
        self.neighbor_matrix = self.compute_neighbor_matrix()

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.value_network = ValueNetwork(input_dim, 1)
        self.target_value_network = ValueNetwork(input_dim, 1)
        self.policy_network = PolicyNetwork(input_dim, output_dim)

        # At the very beginning the value network and target value network should be equivalent
        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(param)

        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        self.MSE_loss = nn.MSELoss()

    def update_learning_rate(self, episode, max_episodes):
        new_learning_rate = self.learning_rate - (episode * self.learning_rate/max_episodes)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=new_learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=new_learning_rate)

    def compute_neighbor_matrix(self):
        neighbor_matrix = np.ndarray((251, 6))  # 251 hex bins and 6 neighbors per bin
        for hex_bin in range(len(self.hex_attr_df)):
            neighbor_matrix[hex_bin] = self.hex_attr_df.iloc[hex_bin].values[1:]
        return neighbor_matrix

    def get_action(self, hex_bin, state):
        state = torch.FloatTensor(state).to(self.device)
        logits = self.policy_network.forward(state)
        for i in range(0, 6):
            if np.isnan(self.neighbor_matrix[hex_bin][i]):
                logits[i] = 0
        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)

        return probs.sample().cpu().detach().item()

    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)

        # Compute value target
        discounted_rewards = [torch.sum(
            torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))])
            * rewards[j:]) for j in range(rewards.size(0))]
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

        # Compute value loss
        values = self.value_network.forward(states)
        value_loss = F.mse_loss(values, value_targets.detach())

        # Compute policy loss with entropy bonus
        logits = self.policy_network.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)

        # Compute entropy bonus
        # entropy = probs.entropy().sum()

        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()
        # policy_loss -= (0.01 * entropy)
        # self.logger.info("Policy loss: {} Entropy contrib: {}".format(policy_loss, 0.01 * entropy))

        return value_loss, policy_loss

    def update(self, batch_size):
        # batch = self.trajectory_buffer.sample_sequences(12, batch_size)
        batch = self.trajectory_buffer.sample(batch_size)

        value_losses = []
        policy_losses = []
        for i in range(len(batch)):
            hex_trajectory = batch[i]
            value_loss, policy_loss = self.compute_loss(hex_trajectory)
            value_losses.append(value_loss)
            policy_losses.append(policy_loss)

        value_loss = torch.stack(value_losses).sum()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.1)
        self.value_optimizer.step()

        policy_loss = torch.stack(policy_losses).sum()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.1)
        self.policy_optimizer.step()
