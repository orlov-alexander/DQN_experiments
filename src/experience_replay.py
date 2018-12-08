import numpy as np


class ExperienceReplay:
    """Experience replay for an off-policy algorithm for environment with discrete action space"""
    def __init__(self, capacity, observation_shape):
        self.capacity = capacity
        self.full = False
        self.cursor = 0

        self.observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.is_done = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.cursor

    def push(self, observations, action, reward, is_done):
        self.observations[self.cursor] = observations
        self.actions[self.cursor] = action
        self.rewards[self.cursor] = reward
        self.is_done[self.cursor] = is_done
        self.cursor += 1
        if self.cursor >= self.capacity:
            self.cursor = 0
            self.full = True

    def sample(self, batch_size):
        indices = np.random.randint(0, self.__len__() - 1, size=batch_size)
        observations = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_observations = self.observations[indices + 1]
        is_done = self.is_done[indices]
        return observations, actions, rewards, next_observations, is_done
