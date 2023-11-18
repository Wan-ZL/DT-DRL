'''
Project     : DT-DRL 
File        : discrete_cartpole_env.py
Author      : Zelin Wan
Date        : 11/7/23
Description : Make cartpole gym environment discrete. For example, make cartpole's cart position [-2, -1, 0, 1, 2]
'''

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv


class CustomCartPoleEnv(CartPoleEnv):  # inherit from Gym CartPoleEnv
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_name = 'CustomCartPole'
        self.discrete_version = 0  # choose from [0, 1, 2, 3, 4, 5, 6, 7, 8]. 0 means original continuous environment.

        if self.discrete_version == 0:
            return

        # Dynamically call the corresponding initialization method
        self.observation_discretization = {}  # Initialize an empty dictionary
        init_method_name = f'discrete_init_ver_{self.discrete_version}'
        print("init_method_name: ", init_method_name)
        if hasattr(self, init_method_name):
            getattr(self, init_method_name)()
        else:
            raise Exception('Invalid discrete environment version')

        # change observation space to discrete
        self.observation_space = gym.spaces.MultiDiscrete([len(discrete_array) for discrete_array in self.observation_discretization.values()])

    # ================== Below are different versions of discretization (begin) ==================
    def discrete_init_ver_1(self):
        '''
        This environment version discretize all 4 observations. Discrete 1 or 0.1 each observation.
        :return:
        '''
        self.cart_position_discret = np.array([-2, -1, 0, 1, 2])
        self.cart_velocity_discret = np.array([-2, -1, 0, 1, 2])
        self.pole_angle_discret = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        self.pole_angle_vel_discret = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        self.all_state_combinations = np.array(np.meshgrid(self.cart_position_discret, self.cart_velocity_discret,
                                                           self.pole_angle_discret,
                                                           self.pole_angle_vel_discret)).T.reshape(-1, 4)
        self.obs_max_value = np.array([2, 2, 0.2, 0.2])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_2(self):
        '''
        This environment version discretize cart position and pole angle only. Discrete 1 or 0.1 each observation.
        :return:
        '''
        self.cart_position_discret = np.array([-2, -1, 0, 1, 2])
        self.pole_angle_discret = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        self.all_state_combinations = np.array(np.meshgrid(self.cart_position_discret,
                                                           self.pole_angle_discret)).T.reshape(-1, 2)
        self.obs_max_value = np.array([2, 0.2])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 2: self.pole_angle_discret}

    def discrete_init_ver_3(self):
        '''
        This environment version discretize all 4 observations. Discrete 0.5 or 0.05 each observation.
        :return:
        '''
        self.cart_position_discret = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        self.cart_velocity_discret = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        self.pole_angle_discret = np.array([-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2])
        self.pole_angle_vel_discret = np.array([-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2])
        self.all_state_combinations = np.array(np.meshgrid(self.cart_position_discret, self.cart_velocity_discret,
                                                           self.pole_angle_discret,
                                                           self.pole_angle_vel_discret)).T.reshape(-1, 4)
        self.obs_max_value = np.array([2, 2, 0.2, 0.2])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_4(self):
        '''
        This environment version discretize cart position and pole angle only. Discrete 0.5 or 0.05 each observation.
        :return:
        '''
        self.cart_position_discret = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        self.pole_angle_discret = np.array([-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2])
        self.all_state_combinations = np.array(np.meshgrid(self.cart_position_discret,
                                                           self.pole_angle_discret)).T.reshape(-1, 2)
        self.obs_max_value = np.array([2, 0.2])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 2: self.pole_angle_discret}

    def discrete_init_ver_5(self):
        '''
        This environment version discretize all 4 observations. Discrete 0.8 or 0.07 each observation.
        :return:
        '''
        self.cart_position_discret = np.array([-2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4])
        self.cart_velocity_discret = np.array([-2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4])
        self.pole_angle_discret = np.array([-0.21, -0.14, -0.07, 0, 0.07, 0.14, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, -0.14, -0.07, 0, 0.07, 0.14, 0.21])
        self.all_state_combinations = np.array(np.meshgrid(self.cart_position_discret, self.cart_velocity_discret,
                                                           self.pole_angle_discret,
                                                           self.pole_angle_vel_discret)).T.reshape(-1, 4)
        self.obs_max_value = np.array([2.4, 2.4, 0.21, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_6(self):
        '''
        This environment version discretize all 4 observations. Discrete 1.2 or 0.105 each observation.
        :return:
        '''
        self.cart_position_discret = np.array([-2.4, -1.2, 0, 1.2, 2.4])
        self.cart_velocity_discret = np.array([-2.4, -1.2, 0, 1.2, 2.4])
        self.pole_angle_discret = np.array([-0.21, -0.105, 0, 0.105, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, -0.105, 0, 0.105, 0.21])
        self.all_state_combinations = np.array(np.meshgrid(self.cart_position_discret, self.cart_velocity_discret,
                                                           self.pole_angle_discret,
                                                           self.pole_angle_vel_discret)).T.reshape(-1, 4)
        self.obs_max_value = np.array([2.4, 2.4, 0.21, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_7(self):
        '''
        This environment version discretize all 4 observations. Discrete 2.4 or 0.21 each observation.
        :return:
        '''
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        self.all_state_combinations = np.array(np.meshgrid(self.cart_position_discret, self.cart_velocity_discret,
                                                           self.pole_angle_discret,
                                                           self.pole_angle_vel_discret)).T.reshape(-1, 4)
        self.obs_max_value = np.array([2.4, 2.4, 0.21, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_8(self):
        '''
        This environment version discretize cart position and pole angle only. Discrete 2.4 or 0.21 each observation.
        :return:
        '''
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.all_state_combinations = np.array(np.meshgrid(self.cart_position_discret,
                                                           self.pole_angle_discret)).T.reshape(-1, 2)
        self.obs_max_value = np.array([2.4, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 2: self.pole_angle_discret}

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

    def step(self, action, **kwargs):
        '''
        Override the step function to discretize the observation
        :param action:
        :return:
        '''
        obs, reward, terminated, truncated, info = super().step(action, **kwargs)   # call the original step function

        # Discretize the observation
        if self.discrete_version != 0:
            obs = self.discrete_obs(obs)

        return obs, reward, terminated, False, info

    def reset(self, **kwargs):
        '''
        Override the reset function to reset the accumulated reward
        :param kwargs:
        :return:
        '''
        return super().reset(**kwargs)


