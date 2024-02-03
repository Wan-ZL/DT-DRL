'''
Project     : DT-DRL 
File        : discrete_cartpole_env.py
Author      : Zelin Wan
Date        : 11/27/23
Description : 
'''
import copy
import gymnasium as gym
import numpy as np
from continuous_cartpole import ContinuousCartPoleEnv


class CustomCartPoleEnv(ContinuousCartPoleEnv):  # inherit from Gym CartPoleEnv
    def __init__(self, env_discrete_version=0, fix_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.env_name = 'CustomCartPole'
        self.discrete_version = env_discrete_version  # choose from [0, 1, 2, 3, 4, 5, 6, 7, 8]. 0 means original continuous environment.
        self.fix_seed = fix_seed    # fix the seed for reproducibility
        self.gravity = 9.81  # change gravity to 9.81
        self.max_step = 500
        self.step_count = 0
        # self.render_mode='human'  # uncomment this line to render the environment

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

        # convert all elements in self.observation_discretization to np.float32
        for key, value in self.observation_discretization.items():
            self.observation_discretization[key] = np.array(value, dtype=np.float32)

        # change action space to discrete
        self.action_space = gym.spaces.Discrete(len(self.action_discrete))
        # change observation space to discrete
        self.observation_space = gym.spaces.MultiDiscrete(
            [len(discrete_array) for discrete_array in self.observation_discretization.values()])

        self.all_state_combinations = np.array(np.meshgrid(*self.observation_discretization.values())).T.reshape(-1,
                                                                                                                 len(self.observation_discretization))
        self.obs_max_value = np.array(
            [np.max(discrete_array) for discrete_array in self.observation_discretization.values()])
        # print("action_space: ", self.action_space)
        # print("observation_space: ", self.observation_space)

    # ================== Below are different versions of discretization (begin) ==================

    def discrete_init_ver_1(self):
        # Discretize the action space
        self.action_discrete = np.array([-3, 3])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_2(self):
        # Discretize the action space
        self.action_discrete = np.array([-5, 5])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_3(self):
        # Discretize the action space
        self.action_discrete = np.array([-7, 7])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_4(self):
        # Discretize the action space
        self.action_discrete = np.array([-10, 10])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_5(self):
        # Discretize the action space
        self.action_discrete = np.array([-3, 0, 3])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_6(self):
        # Discretize the action space
        self.action_discrete = np.array([-5, 0, 5])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_7(self):
        # Discretize the action space
        self.action_discrete = np.array([-7, 0, 7])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_8(self):
        # Discretize the action space
        self.action_discrete = np.array([-10, 0, 10])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_9(self):
        # Discretize the action space
        self.action_discrete = np.array([-6, -2, 2, 6])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_10(self):
        # Discretize the action space
        self.action_discrete = np.array([-9, -3, 3, 9])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_11(self):
        # Discretize the action space
        self.action_discrete = np.array([-10, -6, -2, 2, 6, 10])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}


    def discrete_init_ver_12(self):
        # Discretize the action space
        self.action_discrete = np.array([-5, -3, -1, 1, 3, 5])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_13(self):
        # Discretize the action space
        self.action_discrete = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}

    def discrete_init_ver_14(self):
        # Discretize the action space
        self.action_discrete = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])
        # Discretize the observation space
        self.cart_position_discret = np.array([-2.4, 0, 2.4])
        self.cart_velocity_discret = np.array([-2.4, 0, 2.4])
        self.pole_angle_discret = np.array([-0.21, 0, 0.21])
        self.pole_angle_vel_discret = np.array([-0.21, 0, 0.21])
        # Set up the mapping
        self.observation_discretization = {0: self.cart_position_discret, 1: self.cart_velocity_discret,
                                           2: self.pole_angle_discret, 3: self.pole_angle_vel_discret}






    def discrete_obs(self, obs):
        for i, discret_array in self.observation_discretization.items():
            obs[i] = self.map_to_closest_discrete(obs[i], discret_array)
        return obs

    def mapping_observations(self, obs):
        # map the observation to 0 to n-1. For example, [-1.0, 0.0, 1.0] will be mapped to [0, 1, 2]
        for obs_index, obs_value in enumerate(obs):
            obs[obs_index] = np.where(self.observation_discretization[obs_index] == obs_value)[0][0]
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
        # if action is a numpy array, get the first element
        if isinstance(action, np.ndarray):
            action = action[0]
        # Convert the discrete action to continuous action
        if self.discrete_version != 0:
            action = np.array([self.action_discrete[action]])  # convert the discrete action to continuous action

        obs, reward, terminated, truncated, info = super().step(action, **kwargs)  # call the original step function
        self.step_count += 1
        if self.step_count >= self.max_step:
            terminated = True
            # truncated = True

        # Discretize the observation
        if self.discrete_version != 0:
            obs = self.discrete_obs(obs)
            info['discrete_obs'] = copy.deepcopy(obs)
            obs = self.mapping_observations(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        '''
        Override the reset function to discretize the observation
        :param action:
        :return:
        '''

        obs, info = super().reset(seed=self.fix_seed, **kwargs)
        # print("obs from reset: ", obs)
        self.step_count = 0
        # Discretize the observation
        if self.discrete_version != 0:
            obs = self.discrete_obs(obs)
            info['discrete_obs'] = copy.deepcopy(obs)
            obs = self.mapping_observations(obs)

        return obs, info
