import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, 5)
        )
        self.value = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, observation):
        batch = observation.size(0)
        conv = self.conv(observation).view(batch, -1)
        advantage = self.advantage(conv)
        value = self.value(conv)
        q_value = value + (advantage - advantage.mean(-1, keepdim=True))
        return q_value


class Agent:
    def __init__(self, device, gamma=0.99):
        self.device = device
        self.gamma = gamma

        self.policy_net = DQN()
        self.policy_net.to(device)

        self.target_net = DQN()
        self.target_net.to(device)
        self.update_target()

    def parameters(self):
        return self.policy_net.parameters()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, observation):
        observation = torch.tensor([observation], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.policy_net(observation)
        action = q_values.argmax(-1)[0]
        return action.cpu().item()

    def loss_on_batch(self, batch):
        observations, actions, rewards, next_observations, is_done = batch

        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        is_done = torch.tensor(is_done, dtype=torch.float32, device=self.device)

        batch_size = observations.size(0)
        q_values = self.policy_net(observations)
        q_values_for_actions = q_values[torch.arange(batch_size), actions]
        with torch.no_grad():
            curr_q_values = self.policy_net(next_observations)
            next_q_values = self.target_net(next_observations)
        next_actions = curr_q_values.argmax(-1)
        next_q_values_for_actions = next_q_values[torch.arange(batch_size), next_actions]
        target_q_values = rewards + (1.0 - is_done) * self.gamma * next_q_values_for_actions
        loss = 0.5 * ((q_values_for_actions - target_q_values) ** 2).mean()
        return loss
