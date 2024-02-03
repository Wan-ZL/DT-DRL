'''
Project     : DT-DRL 
File        : DT_PPO.py
Author      : Zelin Wan
Date        : 11/29/23
Description : 
'''
import os

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


# class CustomActorCriticPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#
#     def _build(self, lr_schedule):
#         super()._build(lr_schedule)
#         # add a new layer to the end of the actor network
#         # in_features_size = self.net_arch['pi'][-1]
#         # action_space_size = self.action_net.out_features
#         # self.mlp_extractor.policy_net = self.mlp_extractor.policy_net.append(nn.Linear(in_features_size, action_space_size, bias=True))
#         # Apply custom weight initialization for actor only
#         # self.predefine_weight(self.mlp_extractor.policy_net)
#
#
#
#
#     def predefine_weight(self, network, distribution=None):
#         # for layer in network:
#         #     if type(layer) == nn.Linear:
#         #         print("layer.weight.data: ", layer.weight.data)
#         #         print("layer.bias.data: ", layer.bias.data)
#         target_layer = len(self.net_arch['pi'])
#         layer_count = 0
#         action_space_size = self.action_net.out_features
#         for layer in network:
#             if type(layer) == nn.Linear:
#                 layer_count += 1
#                 if layer_count >= target_layer:
#                     if distribution is not None:
#                         initial_prob_distribution = torch.tensor(distribution, dtype=torch.float32)
#                     else:
#                         initial_prob_distribution = torch.tensor([1.0 / action_space_size] * action_space_size, dtype=torch.float32)
#                         # initial_prob_distribution = torch.tensor([1.0 / self.n_actions] * self.n_actions, dtype=torch.float32)
#
#                     print("initial_prob_distribution", initial_prob_distribution)
#                     # initial_prob_distribution = initial_prob_distribution * len(initial_prob_distribution)
#                     # print("HEU_prob_pytorch", initial_prob_distribution, sum(initial_prob_distribution))
#                     HEU_diagonal_matrix = torch.diag(initial_prob_distribution)
#                     layer.weight.data = HEU_diagonal_matrix

class DecisionTheoryGuidedPPOAgent(PPO):
    def __init__(self, policy, env, fix_seed=None, **kwargs):
        super().__init__(policy, env, **kwargs)

        if fix_seed is None:
            path_seed_name = 'RandomSeed'
        else:
            path_seed_name = 'FixSeed'

        # # append 'self.action_net.out_features' to the end of the actor network
        # net_arch[0]['pi'].append(env.action_space.n)

        # net_arch = [{'pi': [64, 64], 'vf': [64, 64]}]
        # super().__init__(CustomActorCriticPolicy, env, policy_kwargs=dict(net_arch=net_arch), **kwargs)

        # read the action distribution from the file
        data_path = './data/' + path_seed_name + '/' + env.env_name + '/agent_DT/env_discrete_ver_' + str(env.discrete_version)
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
        # self.action_distribution = np.array([0.15, 0.36, 0.36, 0.15])

        self.logits_for_network = self.find_logits_for_softmax(self.action_distribution)

        action_net_weight = self.policy.action_net.weight.data
        action_net_bias = self.policy.action_net.bias.data
        # # customize action_net by make it a 2-layer network (same input and output size as before)
        action_net_input_size = self.policy.action_net.in_features
        action_net_output_size = self.policy.action_net.out_features

        # self.policy.action_net = nn.Sequential(nn.Linear(action_net_input_size, action_net_output_size, bias=True),
        #                                        nn.Linear(action_net_output_size, action_net_output_size, bias=True))

        # set the weight and bias of the new action_net
        # self.policy.action_net[0].weight.data = action_net_weight
        # self.policy.action_net[0].bias.data = action_net_bias
        # set pre-defined weight for the last layer of the actor network
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

                    # initial_prob_distribution = initial_prob_distribution * len(initial_prob_distribution)
                    # print("HEU_prob_pytorch", initial_prob_distribution, sum(initial_prob_distribution))

                    HEU_diagonal_matrix = torch.diag(initial_prob_distribution)
                    layer.weight.data = HEU_diagonal_matrix
                    if layer.bias is not None:
                        # bias also follow the same distribution
                        # layer.bias.data = initial_prob_distribution
                        # make bias all zero
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
