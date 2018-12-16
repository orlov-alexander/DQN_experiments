import torch
from tensorboardX import SummaryWriter

from src.utils import create_envs
from src.experience_replay import ExperienceReplay
from src.trainer import Trainer
from src.q_network import Agent

import gym
import matplotlib.pyplot as plt

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve the CarRacing-v0 with DQN')
    parser.add_argument('--experience-replay', type=int, default=5000, help='size of experience replay')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--nepochs', type=int, default=100, help='number of epocs')
    parser.add_argument('--nsteps', type=int, default=2000, help='number of steps')
    parser.add_argument('--n_jobs', type=int, default=2, help='number of env jobs')
    parser.add_argument('--prefix', type=str, default='t_unk', help='prefix for logs')
    args = parser.parse_args()

    train_env, test_env = create_envs()
    exp_replay = ExperienceReplay(args.experience_replay, train_env.observation_space.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(device)
    #agent.load('logs/t4/epoch_99.pth')
    optimizer = torch.optim.Adam(agent.parameters(), args.learning_rate)
    logdir = 'logs/' + args.prefix

    config_fname = 'logs/' + args.prefix + '.txt'
    with open(config_fname, 'w') as f:
        f.write(json.dumps(vars(args)))

    writer = SummaryWriter(logdir)
    trainer = Trainer(train_env, test_env, 1, exp_replay, agent, optimizer, logdir, writer, 20, n_jobs = args.n_jobs)

    # r = trainer.test_performance()
    # print(r)
    trainer.train(args.nepochs, args.nsteps, args.batch_size, 1)
