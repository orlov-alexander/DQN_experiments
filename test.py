#!/usr/bin/python
import gym
import sys
import os
from gym import wrappers
from pyvirtualdisplay import Display
from tqdm import tqdm

display = Display(visible = 0, size = (900, 900))
display.start()

import torch
from tqdm import trange

from src.utils import EnvWrapper
from src.q_network import Agent

action_dict = {
    0: [-1, 0, 0],
    1: [+1, 0, 0],
    2: [0, 1, 0],
    3: [0, 0, 0.8],
    4: [0, 0, 0]
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = Agent(device)
checkpoint = 'exp_5epoch_95.pth'
agent.load(checkpoint, map_location='cpu')
#agent.load(checkpoint)
agent.eval()

RENDER = False
try:
    test = 'CarRacing-v0'
    #test_env = EnvWrapper(gym.make('CarRacing-v0'), 4, 1)
    #env = gym.make(test)
    env = EnvWrapper(gym.make('CarRacing-v0'), 4, 1)
    env.seed(1442)

    env = wrappers.Monitor(env, '/tmp/{0}-1'.format(test[2:]), force=True)

    counter = 0
    total_score = 0
    cumsum_reward = 0
    epochs_cumsum = []
    for i in range(10):
        done = False
        observation = env.reset()
        score_round = 0
        for j in range(1000):
            tick = counter + 1
            counter = tick
            #action = env.action_space.sample()  # take a random action
            action = action_dict[agent.act(observation)]
            observation, reward, done, info = env.step(action)
            cumsum_reward += reward
            score_round += reward
            if j % 10 == 0:
                #print('Reward: %0.2f. Total: %s. Epoch: %s. Step: %s' % (reward, total_score, i, j)
                pass
            env.render(mode = 'state_array')
            if done and env.unwrapped.tile_visited_count == len(env.unwrapped.track):
                total_score += 1000 - 0.1 * j
                print("Episode {0} finished".format(i + 1), 1000 - 0.1 * j)
                # env.monitor.close()
                break
            elif done:
                epochs_cumsum.append(score_round)
                break
    print('Total score: %s' % total_score)
    print('Cumsum score: %s' % cumsum_reward)
    print('Epoch cumsum score:', epochs_cumsum)
except KeyboardInterrupt:
    sys.exit()
except IndexError:
    print("Error! Missing Test.")
    print("Please run: ")
    print("{0} Test-v0".format(sys.argv[0]))
    sys.exit()
except gym.error.UnregisteredEnv:
    print("Test not found!")
    sys.exit()
except gym.error.UnsupportedMode:
    print("This doesnt support render mode!")
