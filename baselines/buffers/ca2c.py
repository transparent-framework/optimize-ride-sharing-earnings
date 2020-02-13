import random
from collections import deque


class ValueBuffer(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, global_states, V_target):
        global_states = global_states.detach().cpu().numpy()
        V_target = V_target.detach().cpu().numpy()
        for h in range(251):  # 251 hex bins
            experience = (global_states[h], V_target[h])
            self.buffer.append(experience)

    def sample(self, batch_size):
        global_state_batch = []
        V_target_batch = []
        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            global_state, V_target = experience
            global_state_batch.append(global_state)
            V_target_batch.append(V_target)

        return (global_state_batch, V_target_batch)

    def size(self):
        return len(self.buffer)


class PolicyBuffer(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, global_states, joint_action, policy_embedding, advantage):
        global_states = global_states.detach().cpu().numpy()
        policy_embedding = policy_embedding.detach().cpu().numpy()
        advantage = advantage.detach().cpu().numpy()

        for h in range(251):  # 251 hex bins
            experience = (global_states[h], joint_action[h], policy_embedding[h], advantage[h])
            self.buffer.append(experience)

    def sample(self, batch_size):
        global_state_batch = []
        action_batch = []
        policy_embedding_vector_batch = []
        advantage_batch = []
        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            global_state, action, policy_embedding_vector, advantage = experience
            global_state_batch.append(global_state)
            action_batch.append(action)
            policy_embedding_vector_batch.append(policy_embedding_vector)
            advantage_batch.append(advantage)

        return (global_state_batch, action_batch, policy_embedding_vector_batch, advantage_batch)

    def size(self):
        return len(self.buffer)
