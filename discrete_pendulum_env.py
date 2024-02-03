'''
Project     : DT-DRL 
File        : discrete_pendulum_env.py
Author      : Zelin Wan
Date        : 11/13/23
Description : Make pendulum gym environment discrete.
'''
import numpy as np
import gymnasium as gym
from gymnasium.envs.classic_control import PendulumEnv


class CustomPendulumEnv(PendulumEnv):
    def __init__(self, env_discrete_version=0, fix_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.env_name = 'CustomPendulum'
        self.fix_seed = fix_seed
        self.discrete_version = env_discrete_version  # choose from [0, 1]. 0 means original continuous environment.
        self.g = 9.81  # change gravity to 9.81
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

    # ================== Below are different versions of discretization (begin) ==================

    def discrete_init_ver_1(self):
        # Discretize the action space
        self.action_discrete = np.array([-2.0, 0, 2.0])
        self.action_space = gym.spaces.Discrete(len(self.action_discrete))
        # Discretize the observation space
        self.x_pos_discrete = np.array([-1.0, 0.0, 1.0])
        self.y_pos_discrete = np.array([-1.0, 0.0, 1.0])
        self.ang_vel_discrete = np.array([-4.0, 0.0, 4.0])

        self.obs_max_value = np.array([1.0, 1.0, 8.0])  # the maximum value of each observation (only used by DT agent)
        self.all_state_combinations = np.array(np.meshgrid(self.x_pos_discrete,
                                                           self.y_pos_discrete,
                                                           self.ang_vel_discrete
                                                           )).T.reshape(-1, 3)  # (only used by DT agent))

        self.observation_discretization = {0: self.x_pos_discrete, 1: self.y_pos_discrete, 2: self.ang_vel_discrete}
        self.observation_space = gym.spaces.MultiDiscrete(
            [len(self.x_pos_discrete), len(self.y_pos_discrete), len(self.ang_vel_discrete)])

    def discrete_init_ver_2(self):
        # Discretize the action space
        self.action_discrete = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.action_space = gym.spaces.Discrete(len(self.action_discrete))
        # Discretize the observation space
        self.x_pos_discrete = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.y_pos_discrete = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.ang_vel_discrete = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
        self.obs_max_value = np.array([1.0, 1.0, 8.0])  # the maximum value of each observation (only used by DT agent)
        self.all_state_combinations = np.array(np.meshgrid(self.x_pos_discrete,
                                                           self.y_pos_discrete,
                                                           self.ang_vel_discrete
                                                           )).T.reshape(-1, 3)  # (only used by DT agent))
        self.observation_discretization = {0: self.x_pos_discrete, 1: self.y_pos_discrete, 2: self.ang_vel_discrete}
        self.observation_space = gym.spaces.MultiDiscrete(
            [len(self.x_pos_discrete), len(self.y_pos_discrete), len(self.ang_vel_discrete)])

    def discrete_init_ver_3(self):
        # Discretize the action space
        self.action_discrete = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        self.action_space = gym.spaces.Discrete(len(self.action_discrete))
        # Discretize the observation space
        self.x_pos_discrete = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.y_pos_discrete = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        self.ang_vel_discrete = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
        self.obs_max_value = np.array([1.0, 1.0, 8.0])  # the maximum value of each observation (only used by DT agent)
        self.all_state_combinations = np.array(np.meshgrid(self.x_pos_discrete,
                                                           self.y_pos_discrete,
                                                           self.ang_vel_discrete
                                                           )).T.reshape(-1, 3)  # (only used by DT agent))
        self.observation_discretization = {0: self.x_pos_discrete, 1: self.y_pos_discrete, 2: self.ang_vel_discrete}
        self.observation_space = gym.spaces.MultiDiscrete(
            [len(self.x_pos_discrete), len(self.y_pos_discrete), len(self.ang_vel_discrete)])

    def get_action_torque(self, action_id):
        return self.action_discrete[action_id]

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

    def step(self, action):
        '''
        Override the step function to discretize the observation
        :param action:
        :return:
        '''
        # Convert the discrete action to continuous action
        if self.discrete_version != 0:
            action = np.array([self.action_discrete[action]])  # convert the discrete action to continuous action

        obs, reward, terminated, truncated, info = super().step(action)  # call the original step function
        self.step_count += 1
        if self.step_count >= self.max_step:
            terminated = True
            truncated = True

        # Discretize the observation
        if self.discrete_version != 0:
            obs = self.discrete_obs(obs)
            info['discrete_obs'] = obs
            obs = self.mapping_observations(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        '''
        Override the reset function to discretize the observation
        :param kwargs:
        :return:
        '''
        obs, info = super().reset(seed=self.fix_seed, **kwargs)
        self.step_count = 0
        # Discretize the observation
        if self.discrete_version != 0:
            obs = self.discrete_obs(obs)
            obs = self.mapping_observations(obs)

        return obs, info
