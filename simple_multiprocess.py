import sys
import importlib
from src.car_env import CarRacing
env =



import gym
from src.utils import EnvWrapper
from pyvirtualdisplay import Display
from tqdm import tqdm
import numpy as np
from pathos import multiprocessing
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def get_fully_wrapped_env(wrapper_class=EnvWrapper, wrapper_params={},
                          n_jobs=2, env_name='CarRacing-v0'):
    def make_env(rank, seed=0):
        # def _init():
        display = Display(visible=0, size=(900, 900))
        display.start()
        env = gym.make(env_name)
        env.seed(seed + rank)
        env = wrapper_class(env, n_frames=4, frame_skip=1)
        return env

        # return _init

    env = [make_env(i) for i in range(n_jobs)]
    #env = SubprocVecEnv([make_env(i) for i in range(n_jobs)])
    return env

print('kek')
envs = get_fully_wrapped_env(n_jobs = 2)
env = envs[0]
for env in envs:
    env.reset()
for env in tqdm(envs):
    action = env.action_space.sample()
    a = env.step(action)

from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from src.utils import create_envs
from src.experience_replay import ExperienceReplay
from src.trainer import Trainer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
from src.noisy_linear_layer import NoisyLinear
from src.q_network import DQN
from src.q_network import Agent



agent = Agent(device)
agent.load('exp_5epoch_95.pth')

action = env.action_space.sample()
a = env.step(action)
act = np.concatenate([a[0][np.newaxis, ...]] * 10)
act.shape
agent.act(act)

n_jobs = 2

def step_1(env):
    return env.step(action)

pool = multiprocessing.ThreadPool(processes=n_jobs)
res = list(map(step_1, envs))
res = [step_1(env) for env in envs]
res = pool.map(step_1, envs)
# pool.close()
#env.action_space.sample()
for _ in tqdm(range(1000)):
    agent.act(act)
# for _ in tqdm(range(1000)):
#     action = env.action_space.sample()
#     a = env.step(action)