from tqdm import tqdm
# TODO: write Normal, OU noises


class Trainer:
    def __init__(self,
                 train_environment, test_environment, num_tests, experience_replay,
                 agent, actor_optimizer, critic_optimizer,
                 writer, write_frequency):
        # environments
        self.train_environment = train_environment
        self.test_environment = test_environment
        self.num_tests = num_tests
        self.experience_replay = experience_replay

        # agent and optimizers
        self.agent = agent
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # writer
        self.writer = writer
        self.write_frequency = write_frequency

    def train_step(self, batch):
        critic_loss, actor_loss = self.agent.loss_on_batch(batch)
        # order of optimizing is crucial! It needs to be checked
        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def test_performance(self):
        """just runs current agent in test environment until end
         and returns full episode reward
         """
        observation = self.test_environment.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = self.agent.act(observation)
            observation, reward, done, _ = self.test_environment.step(action)
            episode_reward += reward
        return episode_reward

    def train(self, num_epochs, num_steps, batch_size):
        """main function of the class

        :param num_epochs: number of training epochs, agent will be tested at every epoch end
        :param num_steps: number of training steps per epoch
        :param batch_size:
        :return:
        """
        # test performance before training
        test_reward = sum([self.test_performance() for _ in range(self.num_tests)])
        self.writer.add_scalar('test_reward', test_reward / self.num_tests, 0)

        # TODO: schedule for noise based on num_steps
        observation = self.train_environment.reset()
        mean_actor_loss, mean_critic_loss, mean_reward = 0.0, 0.0, 0.0
        for epoch in range(num_epochs):
            # TODO: настроить ncols, сделать так, чтобы эпохи 0-9 выводились как 0x
            for step in tqdm(range(num_steps), desc='epoch_{}'.format(epoch+1), ncols=80):
                # add new experience
                action = self.agent.act(observation)  # TODO: add noise here?
                new_observation, reward, done, _ = self.train_environment.step(action)
                self.experience_replay.push(observation, action, reward, done)
                # sample some experience for training
                batch = self.experience_replay.sample(batch_size)
                critic_loss, actor_loss = self.train_step(batch)

                # update writer statistics
                mean_actor_loss += actor_loss
                mean_critic_loss += critic_loss
                mean_reward += reward

                # write logs
                if (epoch * num_steps + step) % self.write_frequency == 0:
                    d = self.write_frequency  # 'd' stands for 'denominator'
                    log_step = (epoch * num_steps + step) // d
                    self.writer.add_scalar('critic_loss', mean_critic_loss / d, log_step)
                    self.writer.add_scalar('actor_loss', mean_actor_loss / d, log_step)
                    self.writer.add_scalar('batch_reward', mean_reward / d, log_step)
                    mean_actor_loss, mean_critic_loss, mean_reward = 0.0, 0.0, 0.0

                # update current observation
                if done:
                    observation = self.train_environment.reset()
                else:
                    observation = new_observation
            test_reward = sum([self.test_performance() for _ in range(self.num_tests)])
            self.writer.add_scalar('test_reward', test_reward / self.num_tests, epoch + 1)
