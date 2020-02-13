import logging
import torch
import numpy as np
from tqdm import tqdm
from agents.ca2c import cA2CAgent
import gym
import gym_nyc_yellow_taxi


class cA2C(object):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("baseline_logger")

        # cA2C config
        self.baseline_config = self.config['cA2C_parameters']

        # Create environment
        env_id = "nyc-yellow-taxi-v0"
        self.env = gym.make(env_id, config_=self.config)

        # Num hex bins and total time steps
        self.S = self.env.S
        self.T = self.env.T

    def create_one_hot_hex_matrix(self, neighbor_matrix):
        hex_matrix = np.zeros((251, 251), dtype=int)
        for i in range(251):
            for j in range(6):
                if not np.isnan(neighbor_matrix[i][j]):
                    neighbor = neighbor_matrix[i][j]
                    hex_matrix[i][int(neighbor)] = 1
        return hex_matrix

    def mini_batch_train(self, env, agent, city_states, max_episodes, max_steps, max_update_steps, batch_size):
        episode_rewards = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        one_hot_hex_matrix = np.eye(self.S)
        one_hot_step_matrix = np.eye(self.T)
        # one_hot_hex_matrix = np.zeros((self.S, self.S))
        # one_hot_neighbor_matrix = self.create_one_hot_hex_matrix(agent.neighbor_matrix)
        # one_hot_step_matrix = np.zeros((self.T, 9))
        # for h in range(self.S):
        #     one_hot_hex_matrix[h] = one_hot_neighbor_matrix[h].flatten()
        # for s in range(self.T):
        #     one_hot_step_matrix[s] = np.array(list(np.binary_repr(s, width=9)), dtype=int)

        for episode in range(max_episodes):
            env_state = env.reset()
            env_state = np.concatenate(env_state, axis=None)
            episode_reward = 0

            for step in range(max_steps):
                # For each hex bin create state vector
                env_action = []
                city_state = city_states[step]['pickup_vector']
                global_states = np.array([np.concatenate((one_hot_hex_matrix[h], env_state, city_state,
                                                          one_hot_step_matrix[step]),
                                         axis=None) for h in range(self.S)])
                global_states = torch.FloatTensor(global_states).to(device)

                global_vals = agent.value_network.forward(global_states)
                logit_vals = agent.policy_network.forward(global_states)

                prob_valid_action_matrix, policy_embedding, env_action = agent.compute_joint_action(
                    global_states, global_vals, logit_vals)

                env_next_state, env_rewards, done, _ = env.step(env_action)
                env_rewards = torch.FloatTensor(env_rewards).to(device)

                # Populate replay buffer transitions for each hex bin
                env_next_state = env_next_state
                env_next_state = np.concatenate(env_next_state, axis=None)

                next_step = (step + 1) % self.T

                next_city_state = city_states[next_step]['pickup_vector']

                global_next_states = np.array([np.concatenate((one_hot_hex_matrix[h], env_next_state, next_city_state,
                                                               one_hot_step_matrix[next_step]),
                                              axis=None) for h in range(self.S)])
                global_next_states = torch.FloatTensor(global_next_states).to(device)
                global_target_vals = agent.target_value_network.forward(global_next_states)
                discounted_target_vals = (agent.gamma * global_target_vals) + env_rewards.unsqueeze(1)

                V_target = torch.FloatTensor([
                    torch.sum(logit_vals[h] * discounted_target_vals[h]).item() for h in range(self.S)]).to(device)

                # Add to value buffer
                agent.value_buffer.push(global_states, V_target)

                # Add to policy buffer
                advantage = discounted_target_vals - global_vals
                agent.policy_buffer.push(global_states, env_action, policy_embedding, advantage)

                episode_reward += np.sum(env_rewards.detach().cpu().numpy())

                if done:
                    break

                env_state = env_next_state

            print("Episode: {} Reward: {}\n".format(episode, episode_reward))

            # Update networks
            for _ in tqdm(range(max_update_steps)):
                agent.update(batch_size)

            # Update target value network
            for target_param, param in zip(agent.target_value_network.parameters(), agent.value_network.parameters()):
                target_param.data.copy_(agent.tau * param + (1 - agent.tau) * target_param)

            episode_rewards.append(episode_reward)

        return episode_rewards

    def run(self, city_states):

        learning_rate = self.baseline_config['learning_rate']
        gamma = self.baseline_config['gamma']
        tau = self.baseline_config['tau']
        max_episodes = self.baseline_config['max_episodes']
        max_timesteps = self.baseline_config['max_timesteps']
        max_update_steps = self.baseline_config['max_update_steps']
        batch_size = self.baseline_config['batch_size']
        buffer_size = self.baseline_config['buffer_size']

        self.env.reset()
        # input_dim = 1041 # 3 * 251 + 288
        # output_dim = 7 (6 neighbors to relocate + 1 to wait in same hex bin)
        agent = cA2CAgent(self.env, learning_rate=learning_rate, gamma=gamma, tau=tau, buffer_size=buffer_size,
                          input_dim=1041, output_dim=7)
        episode_rewards = self.mini_batch_train(self.env, agent, city_states, max_episodes, max_timesteps,
                                                max_update_steps, batch_size)

        return episode_rewards
