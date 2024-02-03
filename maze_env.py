'''
Project     : DT-DRL 
File        : maze_env.py
Author      : Zelin Wan
Date        : 12/17/23
Description : State: [x, y], x: horizontal, y: vertical. Action: 0: up, 1: down, 2: right, 3: left
'''

import copy
import gymnasium as gym
import numpy as np
import random

from gym_maze_2.envs.maze_env import MazeEnv

class CustomMaze(MazeEnv):
    def __init__(self, env_discrete_version=0, fix_seed=None, maze_file=None, maze_size=None, mode=None, enable_render=False):
        self.env_name = 'CustomMaze'
        self.discrete_version = env_discrete_version
        if maze_size is None:
            maze_size = (self.discrete_version, self.discrete_version)
        else:
            print("maze_size is given as: ", maze_size, ". discrete_version will be ignored.")
        print("maze_size: ", maze_size)
        self.start_state = (0, 0)
        self.goal_state = (maze_size[0] - 1, maze_size[1] - 1)
        self.maze_size = maze_size
        self.fix_seed = fix_seed  # fix the seed for reproducibility
        if self.fix_seed is not None:
            random.seed(self.fix_seed)
        self.step_count = 0
        self.max_step = 10 * maze_size[0] * maze_size[1]

        super().__init__(maze_file=maze_file, maze_size=maze_size, mode=mode, enable_render=enable_render)

        if enable_render:
            self.render()



    def step(self, action, **kwargs):
        self.step_count += 1

        # if action is in np.array, get the first element
        if type(action) == np.ndarray:
            action = action[0]

        if type(action) != np.int64:
            action = np.int64(action)

        obs, reward, terminated, info = super().step(action, **kwargs)
        if self.step_count >= self.max_step:
            terminated = True
            truncated = True
        else:
            truncated = False

        if self.enable_render:
            self.render()

        # make obs integer
        obs = np.array(obs).astype(int)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        if self.fix_seed is not None:
            np.random.seed(self.fix_seed)

        self.step_count = 0

        obs = super().reset(**kwargs)
        info = {}

        obs = np.array(obs).astype(int)
        return obs, info

