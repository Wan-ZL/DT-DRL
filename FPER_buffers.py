'''
Project     : DT-DRL 
File        : FPER_buffers.py
Author      : Zelin Wan
Date        : 1/31/24
Description : Customize the RolloutBuffer to use the Freshness Prioritized Experience Replay (FPER) method
'''
import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces
from typing import Any, Dict, Generator, List, Optional, Union
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize


class FPERRolloutBuffer(RolloutBuffer):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.replayed_count = np.zeros(self.buffer_size * self.n_envs)    # the number of times each sample has been replayed
        # the TD errors of each sample. Will be used to calculate the priority. TD_error initial with big value
        self.TD_errors = np.ones(self.buffer_size * self.n_envs) * 1000
        self.epsilon = 1.0 # gradually decrease the epsilon to 0.1
        self.refresh_factor = 0.9
        self.reset()
        print("FPERRolloutBuffer initialized")


    def get_prob_of_replay(self):
        # calculate the priority of each sample
        # priority = np.abs(self.TD_errors) # + self.epsilon
        # priority[:100] += 1000  # add a large value for testing
        priority = np.abs(self.TD_errors) * self.refresh_factor ** self.replayed_count + self.epsilon
        self.epsilon -= 0.01 if self.epsilon > 0.1 else 0

        # calculate the probability of each sample being replayed
        prob = priority / np.sum(priority)
        return prob

    def update_replayed_count(self, replayed_indices):
        self.replayed_count[replayed_indices] += 1

    def update_TD_errors(self, TD_errors, replayed_indices):
        # update the TD errors only for the samples that are being replayed
        self.TD_errors[replayed_indices] = TD_errors

    def get_old(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size]), indices[start_idx : start_idx + batch_size]
            start_idx += batch_size

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""

        # create a indeces array of size buffer_size * n_envs
        indices = np.arange(self.buffer_size * self.n_envs)

        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        replayed_idx_count = 0
        while replayed_idx_count < self.buffer_size * self.n_envs:
            # Calculate probabilities of each sample being replayed
            prob = self.get_prob_of_replay()
            # selection batch_size number of samples based on the probability
            replayed_indices = np.random.choice(indices, batch_size, p=prob)
            yield self._get_samples(replayed_indices), replayed_indices
            replayed_idx_count += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def reset(self) -> None:
        super().reset()
        # reset the replayed_count, TD_errors and epsilon
        self.replayed_count = np.zeros(self.buffer_size * self.n_envs)  # the number of times each sample has been replayed
        self.TD_errors = np.ones(self.buffer_size * self.n_envs) * 1000
        self.epsilon = 1.0  # gradually decrease the epsilon to 0.1


