'''
Project     : DT-DRL 
File        : discrete_pendulum_env.py
Author      : Zelin Wan
Date        : 11/13/23
Description : Make pendulum gym environment discrete.
'''

from gymnasium.envs.classic_control import PendulumEnv
import numpy as np

class CustomPendulumEnv(PendulumEnv):
    def __init__(self):
        super().__init__()
        self.env_name = 'Pendulum'
        self.discrete_version = 1
        self.g = 9.81   # change gravity to 9.81
        # Dynamically call the corresponding initialization method
        self.observation_discretization = {}  # Initialize an empty dictionary
        init_method_name = f'discrete_init_ver_{self.discrete_version}'
        print("init_method_name: ", init_method_name)
        if hasattr(self, init_method_name):
            getattr(self, init_method_name)()
        else:
            raise Exception('Invalid discrete environment version')

    # ================== Below are different versions of discretization (begin) ==================
    def discrete_init_ver_1(self):
        pass

    def discrete_obs(self, obs):
        for i, discret_array in self.observation_discretization.items():
            obs[i] = self.map_to_closest_discrete(obs[i], discret_array)
        return obs

    # ========================== (end) ==========================

    def map_to_closest_discrete(self, continuous_val, discrete_val_array):
        '''
        Find the discrete value that is closest to the continuous value
        :param discrete_val_array:
        :return:
        '''
        # Find the difference between the continuous value and each discrete value
        differences = np.abs(discrete_val_array - continuous_val)
        # Find the index of the smallest difference
        index_of_smallest_difference = np.argmin(differences)
        # Return the discrete value that is closest to the continuous value
        return discrete_val_array[index_of_smallest_difference]

    def step(self, action):
        '''
        Override the step function to discretize the observation
        :param action:
        :return:
        '''
        obs, reward, terminated, truncated, info = super().step(action)  # call the original step function
        # Discretize the observation
        obs = self.discrete_obs(obs)

        return obs, reward, terminated, truncated, info

