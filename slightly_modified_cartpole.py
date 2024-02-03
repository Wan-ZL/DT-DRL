'''
Project     : DT-DRL 
File        : slightly_modified_cartpole.py
Author      : Zelin Wan
Date        : 1/22/24
Description : Following the original design (like, observation space, action space, etc.), slightly modify the CartPole environment to fit
'''

from gymnasium.envs.classic_control import CartPoleEnv
import numpy as np

class SlightlyModifiedCartPoleEnv(CartPoleEnv):  # inherit from Gym CartPoleEnv
    def __init__(self, env_discrete_version=1, fix_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.env_name = 'SlightlyModifiedCartPole'
        self.discrete_version = env_discrete_version  # Always 0 for this environment
        self.fix_seed = fix_seed    # fix the seed for reproducibility
        self.gravity = 9.81  # change gravity to 9.81
        self.max_step = env_discrete_version * 100 #100 # discrete version associated with the max_step
        self.step_count = 0


    def step(self, action, **kwargs):
        if isinstance(action, np.ndarray) and action.shape != ():
            action = action[0]

        obs, reward, terminated, truncated, info = super().step(action, **kwargs)  # call the original step function
        self.step_count += 1
        if self.step_count >= self.max_step:
            terminated = True
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        obs, info = super().reset(seed=self.fix_seed, **kwargs)
        # print("obs from reset: ", obs)
        self.step_count = 0

        return obs, info