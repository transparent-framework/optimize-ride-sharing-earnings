import logging
import numpy as np
from tqdm import tqdm
from agents.cdqn import cDQNAgent
import gym
import gym_nyc_yellow_taxi


class cDQN(object):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("baseline_logger")

        # cDQN config
        self.baseline_config = self.config['cDQN_parameters']

        # Create environment
        env_id = "nyc-yellow-taxi-v0"
        self.env = gym.make(env_id, config_=self.config)

        # Num hex bins and total time steps
        self.S = self.env.S
        self.T = self.env.T

    def mini_batch_train(self, env, agent, city_states, max_episodes, max_steps, max_update_steps, batch_size):
        episode_rewards = []
        one_hot_step_matrix = np.eye(self.T)
        one_hot_hex_matrix = np.eye(self.S)

        for episode in range(max_episodes):
            # greedy_epsilon = 0.5 * np.e**(-0.1 * episode)
            greedy_epsilon = 0.5 - (episode * 0.5/max_episodes)
            if greedy_epsilon < 0.1:
                greedy_epsilon = 0.1
            env_state = np.concatenate(env.reset(), axis=None)
            episode_reward = 0

            for step in range(max_steps):
                # For each hex bin create state vector
                env_action = []
                city_state = city_states[step]['pickup_vector']
                step_vector = one_hot_step_matrix[step]

                # Iterate over 251 hex bins
                for h in range(self.S):
                    hex_bin_vector = one_hot_hex_matrix[h]
                    state = np.concatenate((hex_bin_vector, env_state, city_state, step_vector), axis=None)

                    # Get action for each hex_bin
                    action = agent.get_action(h, state, greedy_epsilon)
                    env_action.append(action)

                env_next_state, env_rewards, done, _ = env.step(env_action)

                # Populate replay buffer transitions for each hex bin
                env_next_state = np.concatenate(env_next_state, axis=None)
                next_step = (step + 1) % self.T

                next_city_state = city_states[next_step]['pickup_vector']
                next_step_vector = one_hot_step_matrix[next_step]

                for h in range(self.S):
                    hex_bin_vector = one_hot_hex_matrix[h]
                    state = np.concatenate((hex_bin_vector, env_state, city_state, step_vector), axis=None)
                    next_state = np.concatenate((hex_bin_vector, env_next_state, next_city_state, next_step_vector),
                                                axis=None)
                    agent.replay_buffer.push(state, env_action[h], env_rewards[h], next_state, done)

                # Get ready for next step
                env_state = env_next_state
                episode_reward += np.sum(env_rewards)

            episode_rewards.append(episode_reward)
            print("Episode: {} Reward: {}\n".format(episode, episode_reward))

            # Update model
            print("Updating model\n")
            for k in tqdm(range(max_update_steps)):
                agent.update(batch_size)

        return episode_rewards

    def run(self, city_states):

        learning_rate = self.baseline_config['learning_rate']
        gamma = self.baseline_config['gamma']
        max_episodes = self.baseline_config['max_episodes']
        max_timesteps = self.baseline_config['max_timesteps']
        max_update_steps = self.baseline_config['max_update_steps']
        batch_size = self.baseline_config['batch_size']
        buffer_size = self.baseline_config['buffer_size']

        self.env.reset()
        # input_dim = 1041 # 3 * 251 + 288
        # output_dim = 7 (6 neighbors to relocate + 1 to wait in same hex bin)
        agent = cDQNAgent(self.env, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size,
                          input_dim=1041, output_dim=7)
        episode_rewards = self.mini_batch_train(self.env, agent, city_states, max_episodes, max_timesteps,
                                                max_update_steps, batch_size)

        return episode_rewards
