import numpy as np


class RandomAgent:
    def __init__(self):
        pass

    @staticmethod
    def act(observation):
        # call actor network here
        batch_size = observation.shape[0]
        return np.random.normal(size=(batch_size, 3))

    @staticmethod
    def loss_on_batch(batch):
        # call actor and critic networks here, compute losses
        critic_loss = 0
        actor_loss = 0
        return critic_loss, actor_loss
