'''
Project     : DT-DRL
File        : discrete_acrobot_env.py
Author      : Zelin Wan
Date        : 11/12/23
Description : Make acrobot gym environment discrete.
'''

from gymnasium.envs.classic_control import AcrobotEnv
import numpy as np

class CustomCartPoleEnv(AcrobotEnv):  # inherit from Gym AcrobotEnv
    def __init__(self):
        super().__init__()
        self.env_name = 'Acrobot'
        self.discrete_version = 1
        self.observation_space = np.array([0, 0, 0, 0, 0, 0])
        self.action_space = np.array([0, 1, 2])