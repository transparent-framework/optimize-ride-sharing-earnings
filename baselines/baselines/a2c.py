import logging
import numpy as np
from tqdm import tqdm
from agents.a2c import A2CAgent
import gym
import gym_nyc_yellow_taxi


class A2C(object):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("baseline_logger")

        # A2C config
        self.baseline_config = self.config['A2C_parameters']

        # Create environment
        env_id = "nyc-yellow-taxi-v0"
        self.env = gym.make(env_id, config_=self.config)

        # Num hex bins and total time steps
        self.S = self.env.S
        self.T = self.env.T

    def create_one_hot_hex_matrix(self, neighbor_matrix):
        hex_matrix = np.zeros((251, 251), dtype=int)
        for i in range(251):
            # hex_matrix[i][i] = 1
            for j in range(6):
                if not np.isnan(neighbor_matrix[i][j]):
                    neighbor = neighbor_matrix[i][j]
                    hex_matrix[i][int(neighbor)] = 1
        return hex_matrix

    def mini_batch_train(self, env, agent, city_states, max_episodes, max_steps, max_update_steps, batch_size):
        episode_rewards = []
        env_trajectory = [[] for i in range(self.S)]

        # one_hot_hex_matrix = np.eye(self.S)
        # one_hot_step_matrix = np.eye(self.T)
        one_hot_hex_matrix = np.zeros((self.S, self.S))
        one_hot_neighbor_matrix = self.create_one_hot_hex_matrix(agent.neighbor_matrix)
        one_hot_step_matrix = np.zeros((self.T, 9))
        for h in range(self.S):
            one_hot_hex_matrix[h] = one_hot_neighbor_matrix[h].flatten()
        for s in range(self.T):
            one_hot_step_matrix[s] = np.array(list(np.binary_repr(s, width=9)), dtype=int)

        for episode in range(max_episodes):
            env_state = env.reset()
            env_state = np.concatenate(env_state, axis=None)
            env_trajectory = [[] for i in range(self.S)]
            episode_reward = 0
            for step in range(max_steps):
                step_vector = one_hot_step_matrix[step]
                joint_action = []
                hex_states = []
                city_state = city_states[step]['pickup_vector']
                for h in range(self.S):
                    hex_bin_vector = one_hot_hex_matrix[h]
                    hex_state = np.concatenate((step_vector, env_state, hex_bin_vector, city_state), axis=None)
                    hex_states.append(hex_state)
                    action = agent.get_action(h, hex_state)

                    joint_action.append(action)

                next_step = (step + 1) % self.T
                next_step_vector = one_hot_step_matrix[next_step]
                env_next_state, env_rewards, done, _ = env.step(joint_action)
                env_next_state = np.concatenate(env_next_state, axis=None)
                next_city_state = city_states[next_step]['pickup_vector']

                for h in range(self.S):
                    hex_bin_vector = one_hot_hex_matrix[h]
                    next_hex_state = np.concatenate((next_step_vector, env_next_state, hex_bin_vector,
                                                     next_city_state), axis=None)
                    env_trajectory[h].append([hex_states[h], joint_action[h], env_rewards[h], next_hex_state, done])

                agent.trajectory_buffer.push(env_trajectory)

                episode_reward += np.sum(env_rewards)

                if done:
                    break

                env_state = env_next_state

            episode_rewards.append(episode_reward)
            print("Episode: {} Reward: {}\n".format(episode, episode_reward))

            # Update target value network
            for target_param, param in zip(agent.target_value_network.parameters(), agent.value_network.parameters()):
                target_param.data.copy_(agent.tau * param + (1 - agent.tau) * target_param)

            for _ in tqdm(range(max_update_steps)):
                agent.update(batch_size)

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
        # input_dim = 762
        # output_dim = 7
        agent = A2CAgent(self.env, learning_rate=learning_rate, gamma=gamma, tau=tau, buffer_size=buffer_size,
                         input_dim=762, output_dim=7)
        episode_rewards = self.mini_batch_train(self.env, agent, city_states, max_episodes, max_timesteps,
                                                max_update_steps, batch_size)

        return episode_rewards
