'''
Project     : DT-DRL 
File        : decision_theory_agent.py
Author      : Zelin Wan
Date        : 11/8/23
Description : The decision theory agent for the cartpole environment
'''

import numpy as np
from collections import defaultdict


class DecisionTheoryCartpoleAgent:
    def __init__(self, env):
        self.env = env
        # Create a dictionary to store the transition count, default value is 2
        self.obs_action_count_dict = defaultdict(lambda: 2)  # input is old_state + action
        self.transition_count_dict = defaultdict(lambda: 2)  # input is old_state + action + new_state

    def update_observation(self, obs, action, obs_):
        '''
        Update the observation
        :param obs: old state
        :param action: action
        :param obs_: new state
        :return:
        '''
        self.update_obs_action_count_dict(obs, action)
        self.update_transition_count_dict(obs, action, obs_)

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

    def update_transition_count_dict(self, obs, action, obs_):
        '''
        Update the transition count dictionary
        :param obs:
        :param action:
        :param obs_:
        :return:
        '''
        # Update the transition count dictionary
        temp_key = (tuple(obs), action, tuple(obs_))
        self.transition_count_dict[temp_key] += 1

    def utility_function(self, obs):
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

    def expected_utility_function(self, obs, action):
        '''
        Calculate the expected utility. Following equation: EU(s, a) = \sum_{s'} P(s'|s, a) \times U(s')
        :param obs: old state
        :param action:
        :return:
        '''
        expected_utility = 0
        for new_state in self.env.all_state_combinations:
            state_prob = self.transition_count_dict[(tuple(obs), action, tuple(new_state))] / \
                         self.obs_action_count_dict[(tuple(obs), action)]
            state_utility = self.utility_function(new_state)
            expected_utility += state_prob * state_utility
        return expected_utility

    def get_action(self, obs):
        max_utility = 0
        best_action = 0
        for action in range(self.env.action_space.n):
            utility = self.expected_utility_function(obs, action)
            if utility > max_utility:
                max_utility = utility
                best_action = action

        return best_action
