import sys
from importlib import reload
from pyvirtualdisplay import Display
from pathos import multiprocessing

import src.car_env


def get_env():
    reload(src.car_env)
    display = Display(visible = 0, size = (900, 900))
    display.start()
    env = src.car_env.CarRacing()
    env.reset()
    env.step([1, 0, 0])
    return env


envs = [get_env() for _ in range(40)]

def step_1(env):
    return env.step([1, 0, 0])

from tqdm import tqdm
pool = multiprocessing.ThreadPool(processes=2)
res = [step_1(env) for env in tqdm(envs)]
res = pool.map(step_1, envs)
# pool.close()


# from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from src.utils import EnvWrapper
# def get_fully_wrapped_env(wrapper_class=EnvWrapper, wrapper_params={},
#                           n_jobs=2, env_name='CarRacing-v0'):
#     def make_env(rank, seed=0):
#         reload(src.car_env)
#         # def _init():
#         display = Display(visible=0, size=(900, 900))
#         display.start()
#         env = src.car_env.CarRacing()
#         env.seed(seed + rank)
#         #env = wrapper_class(env, n_frames=4, frame_skip=1)
#         return env
#
#         # return _init
#
#     #env = [make_env(i) for i in range(n_jobs)]
#     env = SubprocVecEnv([make_env(i) for i in range(n_jobs)])
#     return env
#
# print('kek')
# envs = get_fully_wrapped_env(n_jobs = 2)