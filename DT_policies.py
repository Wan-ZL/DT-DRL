'''
Project     : DT-DRL 
File        : DT_policies.py
Author      : Zelin Wan
Date        : 1/16/24
Description : modified from stable_baselines3.common.policies.py
'''
import numpy as np
from stable_baselines3.common.policies import *


# modify ActorCriticPolicy
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.DT_agent = None
        # self.utility_list = None
        self.action_prob = None
        self.u_weight = 1.0  # weight for utility when combine with action network output


    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        if self.DT_agent is not None:
            # self.utility_list = np.array([self.DT_agent.get_utility_list(np.array(obs[i])) for i in range(obs.shape[0])])
            self.action_prob = np.array([self.DT_agent.get_DT_action_prob(np.array(obs[i])) for i in range(obs.shape[0])])

        distribution = self._get_action_dist_from_latent(latent_pi, self.action_prob)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        return actions, values, log_prob

    def reverse_softmax(self, input_values):
        """
        Compute the approximate reverse of the softmax function for each row in a 2D array.

        Args:
        input_values (numpy.ndarray): The values after applying softmax, should be a 2D array.

        Returns:
        numpy.ndarray: The approximate values before applying softmax for each row.
        """
        # Ensure input is a numpy array
        values = np.array(input_values)

        # Check if the input is indeed 2D
        if len(values.shape) != 2:
            raise ValueError("Input must be a 2D array.")

        # Replace zeros with a very small number to avoid log(0)
        epsilon = 1e-10
        values_with_epsilon = np.where(values == 0, epsilon, values)

        # Apply the log function to each element
        pre_softmax_values = np.log(values_with_epsilon)

        return pre_softmax_values

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, action_prob=None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        # self.u_weight -= 0.0001 if self.u_weight > 0 else 0
        if action_prob is not None:
            # reverse softmax for utility_list
            reverse_softmax_action_prob = self.reverse_softmax(action_prob)
            reverse_softmax_action_prob = self.u_weight * reverse_softmax_action_prob
            # convert action_prob to tensor
            reverse_softmax_action_prob = th.from_numpy(reverse_softmax_action_prob).float()
            mean_actions = mean_actions + reverse_softmax_action_prob
            self.u_weight -= 0.0001 if self.u_weight > 0 else 0


            # convert utility_list to tensor
            # utility_list = th.from_numpy(utility_list).float()
            #
            # mean_actions = mean_actions + (utility_list * self.u_weight)
            # self.u_weight -= 0.001 if self.u_weight > 0.001 else 0
            # print("self.u_weight: ", self.u_weight)
            # print("mean_actions (after): ", mean_actions)
            # print("softmax(mean_actions): ", th.softmax(mean_actions, dim=1))

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")


    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        if self.DT_agent is not None:
            # self.utility_list = np.array([self.DT_agent.get_utility_list(np.array(obs[i])) for i in range(obs.shape[0])])
            self.action_prob = np.array([self.DT_agent.get_DT_action_prob(np.array(obs[i])) for i in range(obs.shape[0])])
        distribution = self._get_action_dist_from_latent(latent_pi, self.action_prob)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        if self.DT_agent is not None:
            # self.utility_list = np.array([self.DT_agent.get_utility_list(np.array(obs[i])) for i in range(obs.shape[0])])
            self.action_prob = np.array([self.DT_agent.get_DT_action_prob(np.array(obs[i])) for i in range(obs.shape[0])])
        return self._get_action_dist_from_latent(latent_pi, self.action_prob)



