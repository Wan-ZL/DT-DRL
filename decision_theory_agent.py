'''
Project     : DT-DRL 
File        : decision_theory_agent.py
Author      : Zelin Wan
Date        : 11/8/23
Description : The decision theory agent for the cartpole environment
'''

import numpy as np
from collections import defaultdict


class DecisionTheoryAgent:
    def __init__(self, env):
        self.env = env
        # Create a dictionary to store the transition count, default value is 2
        self.obs_action_count_dict = defaultdict(lambda: 2)  # input is old_state + action
        self.transition_count_dict = defaultdict(lambda: 2)  # input is old_state + action + new_state
        # self.obs_reward_list_dict  = defaultdict(lambda: [10])  # input is old_state + action
        self.pre_calc_utility_for_all_states()

    def update_observation(self, obs, action, new_obs, reward):
        '''
        Update the observation
        :param obs: old state
        :param action: action
        :param new_obs: new state
        :param reward: reward
        :return:
        '''
        self.update_obs_action_count_dict(obs, action)
        self.update_transition_count_dict(obs, action, new_obs)
        # self.update_obs_reward_dict(obs, reward)

    # def update_obs_reward_dict(self, obs, reward):
    #     temp_key = tuple(obs)
    #     self.obs_reward_list_dict[temp_key].append(reward)

    def update_obs_action_count_dict(self, obs, action):
        '''
        Update the observation action count dictionary
        :param obs:
        :param action:
        :return:
        '''
        # Update the observation action count dictionary
        temp_key = (tuple(obs), action)  # convert the observation to tuple to be hashable
        self.obs_action_count_dict[temp_key] += 1

    def update_transition_count_dict(self, obs, action, new_obs):
        '''
        Update the transition count dictionary
        :param obs:
        :param action:
        :param new_obs:
        :return:
        '''
        # Update the transition count dictionary
        temp_key = (tuple(obs), action, tuple(new_obs))
        self.transition_count_dict[temp_key] += 1

    def utility_function_for_CustomCartPole(self, obs):
        '''
        Calculate the utility of the observation
        :param obs:
        :return:
        '''
        # Calculate the utility of the observation
        utility = 0
        for i in range(len(obs)):
            utility += 1 - (abs(obs[i]) / self.env.obs_max_value[i])

        # normalize the utility
        utility /= len(obs)

        return utility

    def utility_function_for_CustomPendulum(self, obs, action):
        # Calculate the utility of the observation

        # following Eq. \ref{eq: pendulum-utility-version-1}
        weight = 0.01
        x_preference = obs[0]
        y_preference = abs(obs[1])
        ang_vel_preference = abs(obs[2]) # make angular velocity less impact on the utility
        utility = x_preference - y_preference - (weight * ang_vel_preference)

        # following Eq. \ref{eq: pendulum-utility-version-2}
        # x = obs[0]
        # y = obs[1]
        # ang_vel = obs[2]
        # if x <= 0:
        #     utility = abs(ang_vel)
        # else:
        #     utility = 8 - abs(ang_vel)

        # following equation in pendulum documentation
        # utility = self.utility_reward_dict[(tuple(obs), action)]    # this following the equation \ref{eq: pendulum-utility-version-3}

        return utility

    def pre_calc_utility_for_all_states(self):
        if self.env.env_name == 'CustomPendulum':
            self.utility_reward_dict = defaultdict(lambda: 0)
            for state in self.env.all_state_combinations:
                for action in range(self.env.action_space.n):
                    theta = np.arccos(state[0])
                    self.utility_reward_dict[(tuple(state), action)] = -((theta ** 2) + 0.1 * (state[2] ** 2)) + 0.001 * (self.env.get_action_torque(action) ** 2)
        else:
            print("No need to pre-calculate the utility for this environment")



    def expected_utility_function(self, obs, action):
        '''
        Calculate the expected utility. Following equation: EU(s, a) = \sum_{s'} P(s'|s, a) \times U(s')
        :param obs: old state
        :param action:
        :return:
        '''
        expected_utility = 0
        # print("self.env.all_state_combinations: ", self.env.all_state_combinations)
        for new_state in self.env.all_state_combinations:
            state_prob = self.transition_count_dict[(tuple(obs), action, tuple(new_state))] / \
                         self.obs_action_count_dict[(tuple(obs), action)]

            if self.env.env_name == 'CustomCartPole':
                state_utility = self.utility_function_for_CustomCartPole(new_state)
            elif self.env.env_name == 'CustomPendulum':
                state_utility = self.utility_function_for_CustomPendulum(new_state, action)
            else:
                raise Exception('Unknown environment name')

            expected_utility += state_prob * state_utility
            # expected_utility = state_utility
        return expected_utility

    def get_action(self, obs):
        utility_list = np.array([])
        for action in range(self.env.action_space.n):
            utility = self.expected_utility_function(obs, action)
            utility_list = np.append(utility_list, utility)

        # min-max normalization and convert to probability
        min_utility = min(utility_list)
        max_utility = max(utility_list)
        if min_utility != max_utility:
            # min-max normalization to [0.01, 0.99]
            normalized_utility_list = (utility_list - min_utility) / (max_utility - min_utility) * 0.98 + 0.01
        else:
            # if all the utility are the same, then provide a uniform distribution
            normalized_utility_list = np.ones(len(utility_list))

        action_prob = normalized_utility_list / sum(normalized_utility_list)

        # select one action
        best_action = np.random.choice(self.env.action_space.n, p=action_prob)

        return best_action
