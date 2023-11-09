'''
Project     : DT-DRL 
File        : discret_env.py
Author      : Zelin Wan
Date        : 11/7/23
Description : make gym environment observation discrete. For example, make cartpole's cart position [-2, -1, 0, 1, 2]
'''

import gym
import numpy as np

from gym.envs.classic_control import CartPoleEnv


class CustomCartPoleEnv(CartPoleEnv):   # inherit from Gym CartPoleEnv
    def __init__(self):
        super().__init__()
        self.cart_position_discret = np.array([-2, -1, 0, 1, 2])
        self.cart_velocity_discret = np.array([-2, -1, 0, 1, 2])
        self.pole_angle_discret = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        self.pole_angle_vel_discret = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        self.all_state_combinations = np.array(np.meshgrid(self.cart_position_discret, self.cart_velocity_discret,
                                                     self.pole_angle_discret, self.pole_angle_vel_discret)).T.reshape(-1, 4)
        self.obs_max_value = np.array([2, 2, 0.2, 0.2])

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
        obs, reward, terminated, truncated, info = super().step(action) # call the original step function
        # Discretize the observation
        obs[0] = self.map_to_closest_discrete(obs[0], self.cart_position_discret)
        obs[1] = self.map_to_closest_discrete(obs[1], self.cart_velocity_discret)
        obs[2] = self.map_to_closest_discrete(obs[2], self.pole_angle_discret)
        obs[3] = self.map_to_closest_discrete(obs[3], self.pole_angle_vel_discret)
        return obs, reward, terminated, truncated, info





