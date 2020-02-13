import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from buffers.ca2c import ValueBuffer
from buffers.ca2c import PolicyBuffer
from networks.ca2c import ValueNetwork
from networks.ca2c import PolicyNetwork


class cA2CAgent(object):

    def __init__(self, env, learning_rate, gamma, tau, buffer_size, input_dim, output_dim):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau

        self.value_buffer = ValueBuffer(buffer_size)
        self.policy_buffer = PolicyBuffer(buffer_size)

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

        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        self.MSE_loss = nn.MSELoss()

    def compute_neighbor_matrix(self):
        neighbor_matrix = np.ndarray((251, 6))  # 251 hex bins and 6 neighbors per bin
        for hex_bin in range(len(self.hex_attr_df)):
            neighbor_matrix[hex_bin] = self.hex_attr_df.iloc[hex_bin].values[1:]
        return neighbor_matrix

    def compute_policy_context_embedding(self, global_vals):
        global_vals = global_vals.squeeze(1)
        policy_context_embedding_matrix = np.zeros((251, 7), dtype=int)
        for hex_bin in range(251):
            for i in range(6):
                neighbor = self.neighbor_matrix[hex_bin][i]
                if np.isnan(neighbor):
                    continue
                neighbor = int(neighbor)
                if global_vals[hex_bin] >= global_vals[neighbor]:
                    policy_context_embedding_matrix[hex_bin][i] = 1
            policy_context_embedding_matrix[hex_bin][6] = 1
        return torch.FloatTensor(policy_context_embedding_matrix)

    def compute_prob_valid_action(self, policy_embedding, logit_vals):
        q_valid_matrix = logit_vals * policy_embedding
        prob_valid_action_matrix = F.softmax(q_valid_matrix, dim=1)
        # prob_valid_action_matrix = q_valid_matrix / torch.norm(q_valid_matrix, p=1, dim=1).unsqueeze(1)

        return prob_valid_action_matrix

    def compute_joint_action(self, global_states, global_vals, logit_vals):
        policy_embedding = self.compute_policy_context_embedding(global_vals)
        prob_valid_action_matrix = self.compute_prob_valid_action(policy_embedding, logit_vals)

        try:
            joint_action = [Categorical(prob_valid_action_matrix[h]).sample().cpu().item() for h in range(251)]
        except Exception:
            print('nan values:', torch.sum(torch.isnan(prob_valid_action_matrix)).item())
            raise

        return prob_valid_action_matrix, policy_embedding, joint_action

    def compute_value_loss(self, batch):
        global_states, V_targets = batch
        global_states = torch.FloatTensor(global_states).to(self.device)
        V_targets = torch.FloatTensor(V_targets).to(self.device).unsqueeze(1)

        global_vals = self.value_network.forward(global_states)
        value_loss = self.MSE_loss(V_targets, global_vals)
        return value_loss

    def compute_policy_loss(self, batch):
        global_states, actions, policy_embedding_vectors, advantages = batch
        global_states = torch.FloatTensor(global_states).to(self.device)
        policy_embedding_vectors = torch.FloatTensor(policy_embedding_vectors).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        actions = torch.FloatTensor(actions).to(self.device).unsqueeze(1)

        logit_vals = self.policy_network.forward(global_states)
        logit_vals = logit_vals * policy_embedding_vectors
        logit_vals = F.softmax(logit_vals, dim=1)

        logit_vals = logit_vals / torch.norm(logit_vals, p=1, dim=1).unsqueeze(1)
        probs = Categorical(logit_vals)
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantages.detach()

        policy_loss = policy_loss.mean()
        # entropy = probs.entropy().sum()
        # policy_loss -= (0.01 * entropy)
        return policy_loss

    def update(self, batch_size):

        value_batch = self.value_buffer.sample(batch_size)
        policy_batch = self.policy_buffer.sample(batch_size)
        value_loss = self.compute_value_loss(value_batch)
        policy_loss = self.compute_policy_loss(policy_batch)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 20)
        self.policy_optimizer.step()
