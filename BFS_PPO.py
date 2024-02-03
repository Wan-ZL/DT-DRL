'''
Project     : DT-DRL 
File        : BFS_PPO.py
Author      : Zelin Wan
Date        : 12/21/23
Description : 
'''

import os

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

class BFSGuidedPPOAgent(PPO):
    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)

        # read the action distribution from the file
        data_path = './data/' + env.env_name + '/agent_BFS/env_discrete_ver_' + str(env.discrete_version)
        path_for_action_ratio = data_path + '/last_100_action_ratio/'
        action_ratio_list = []
        if os.path.exists(path_for_action_ratio):
            # read all files in the 'path_for_action_ratio' folder
            files = os.listdir(path_for_action_ratio)
            for file in files:
                file_path = path_for_action_ratio + file
                action_ratio = np.load(file_path)
                action_ratio_list.append(action_ratio)
            self.action_distribution = np.mean(action_ratio_list, axis=0)
        else:
            self.action_distribution = np.array([1.0 / env.action_space.n] * env.action_space.n, dtype=np.float32)

        self.logits_for_network = self.find_logits_for_softmax(self.action_distribution)


        self.predefine_weight_2(self.policy.action_net, self.logits_for_network)

    def predefine_weight(self, network, distribution=None):
        target_layer = 2
        layer_count = 0
        action_space_size = network[-1].out_features
        for layer in network:
            if type(layer) == nn.Linear:
                layer_count += 1
                if layer_count >= target_layer:
                    if distribution is not None:
                        initial_prob_distribution = torch.tensor(distribution, dtype=torch.float32)
                    else:
                        initial_prob_distribution = torch.tensor([1.0 / action_space_size] * action_space_size,
                                                                 dtype=torch.float32)


                    HEU_diagonal_matrix = torch.diag(initial_prob_distribution)
                    layer.weight.data = HEU_diagonal_matrix
                    if layer.bias is not None:
                        layer.bias.data = torch.zeros_like(initial_prob_distribution)

    def predefine_weight_2(self, layer, distribution=None):
        # change the bias to the desired distribution
        layer.bias.data = torch.tensor(distribution, dtype=torch.float32)

    def find_logits_for_softmax(self, desired_distribution, iterations=1000, learning_rate=0.1):
        '''
        Approximate the logits that would produce a given softmax distribution.

        :param desired_distribution: Numpy array, the desired softmax output distribution.
        :param iterations: Number of iterations for the approximation.
        :param learning_rate: Learning rate for the adjustment of logits.
        :return: A numpy array of logits approximating the desired softmax distribution.
        :param desired_distribution:
        :param iterations:
        :param learning_rate:
        :return:
        '''
        # Initialize logits
        logits = np.zeros_like(desired_distribution)

        # Optimization loop
        for _ in range(iterations):
            # Calculate the softmax of the current logits
            e_logits = np.exp(logits)
            softmax = e_logits / np.sum(e_logits)

            # Calculate the gradient of the difference between the current softmax and the desired softmax
            gradient = softmax - desired_distribution

            # Adjust the logits in the direction of the gradient
            logits -= learning_rate * gradient

        return logits
