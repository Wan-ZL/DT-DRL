'''
Project     : DT-DRL 
File        : main.py
Author      : Zelin Wan
Date        : 11/2/23
Description : main function to run all simulations.
'''
import os
import time
import gymnasium as gym
import datetime
import numpy as np
import multiprocessing as mp
from discrete_cartpole_env import CustomCartPoleEnv
from discrete_pendulum_env import CustomPendulumEnv
from decision_theory_agent import DecisionTheoryAgent
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from continuous_cartpole import ContinuousCartPoleEnv


def writer_log_scalar(writer, accumulated_reward, action_count_per_episode, episode, agent_name):
    writer.add_scalar('accumulated_reward/agent: {}'.format(agent_name), accumulated_reward, episode)

    if action_count_per_episode is not None:
        action_ratio = action_count_per_episode / sum(action_count_per_episode)
        for i in range(len(action_ratio)):
            writer.add_scalar('action_ratio/agent: {}, action: {}'.format(agent_name, i), action_ratio[i], episode)
            writer.add_scalar('action_count/agent: {}, action: {}'.format(agent_name, i), action_count_per_episode[i],
                              episode)
    else:
        # for continuous action space, record the action 0 (do nothing)
        writer.add_scalar('action_ratio/agent: {}, action: {}'.format(agent_name, 0), 0, episode)
        writer.add_scalar('action_count/agent: {}, action: {}'.format(agent_name, 0), 0, episode)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting values in tensorboard.
    """

    def __init__(self, env, writer, agent_name, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.writer = writer
        self.agent_name = agent_name
        # Those variables will be accessible in the callback
        self.accumulated_reward = 0
        self.episode_count = 1
        self.action_count_per_episode = np.zeros(self.env.action_space.n) if type(
            self.env.action_space) == gym.spaces.discrete.Discrete else None

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return:
        """
        self.accumulated_reward += self.locals['rewards'][0]
        if self.action_count_per_episode is not None:
            self.action_count_per_episode[self.locals['actions'][0]] += 1

        # check if the episode is done
        episode_done = False
        # check if algorithm is PPO
        if self.model.__class__.__name__ == 'PPO' or self.model.__class__.__name__ == 'A2C':
            episode_done = self.locals.get('done')
        elif self.model.__class__.__name__ == 'DQN':
            episode_done = self.locals.get('dones')[0]

        if episode_done:
            # episode ended, save to tensorboard
            writer_log_scalar(self.writer, self.accumulated_reward, self.action_count_per_episode, self.episode_count,
                              self.agent_name)
            self.accumulated_reward = 0
            self.episode_count += 1
            self.action_count_per_episode = np.zeros(self.env.action_space.n) if type(
                self.env.action_space) == gym.spaces.discrete.Discrete else None
        return True


def run_one_simulation(agent_name, env_discrete_version, env_name, max_episode, max_step):
    # env = gym.make('CartPole-v1', render_mode='human')
    # env = gym.make('CartPole-v1'); env.env_name = 'OrignialCartPole'; env.discrete_version = 0
    if env_name == 'CustomCartPole':
        env = CustomCartPoleEnv(env_discrete_version)
    elif env_name == 'CustomPendulum':
        env = CustomPendulumEnv(env_discrete_version)
    else:
        raise Exception('Invalid environment name')
    env = ContinuousCartPoleEnv()   # this is a test
    env.env_name = 'ContinuousCartPole'
    env.discrete_version = 0
    # env.render_mode = 'human'

    print("env", env)

    # create tensorboard writer
    data_path = './data/' + env.env_name + '/agent_' + agent_name + '/env_discrete_ver_' + str(env.discrete_version)
    # create folder if not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer_path = data_path + '/agent_' + agent_name + '-env_discrete_ver_' + str(
        env.discrete_version) + '-time_' + current_time
    writer = SummaryWriter(writer_path)

    # create agents and models
    DT_agent = None
    DQN_model = None
    PPO_model = None
    A2C_model = None
    if agent_name == 'DT':
        DT_agent = DecisionTheoryAgent(env)
    elif agent_name == 'DQN':
        DQN_model = DQN('MlpPolicy', env, verbose=1)
        DQN_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback(env, writer, agent_name))
        return
    elif agent_name == 'PPO':
        PPO_model = PPO('MlpPolicy', env, verbose=1)
        PPO_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback(env, writer, agent_name))
        return
    elif agent_name == 'A2C':
        A2C_model = PPO('MlpPolicy', env, verbose=1)
        A2C_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback(env, writer, agent_name))
        return
    else:
        pass

    # run the simulation
    reward_history = []
    step_counter = 0
    for episode in range(max_episode):
        obs, info = env.reset()  # fix seed for reproducibility
        terminated = False
        truncated = False
        accumulated_reward = 0
        action_count_per_episode = np.zeros(env.action_space.n)
        while not terminated or not truncated:

            if agent_name == 'DT':
                action = DT_agent.get_action(obs)
            elif agent_name == 'DQN':
                action, _states = DQN_model.predict(obs, deterministic=True)
            elif agent_name == 'PPO':
                action, _states = PPO_model.predict(obs, deterministic=True)
            elif agent_name == 'A2C':
                action, _states = A2C_model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            # count the action
            action_count_per_episode[action] += 1

            new_obs, reward, terminated, truncated, info = env.step(action)

            accumulated_reward += reward

            # DT agent observe the environment and update the belief
            if agent_name == 'DT':
                DT_agent.update_observation(obs, action, new_obs, reward)

            # update the observation
            obs = new_obs

            if terminated or truncated:
                break

        print(f'Episode {episode} ended with accumulated reward {accumulated_reward}')
        # save to tensorboard
        writer.add_scalar('accumulated_reward/agent: {}'.format(agent_name), accumulated_reward, episode)
        action_ratio = action_count_per_episode / sum(action_count_per_episode)
        for i in range(len(action_ratio)):
            writer.add_scalar('action_ratio/agent: {}, action: {}'.format(agent_name, i), action_ratio[i], episode)
            writer.add_scalar('action_count/agent: {}, action: {}'.format(agent_name, i),
                              action_count_per_episode[i],
                              episode)

        reward_history.append(accumulated_reward)

    print(f'Average reward over {max_episode} episodes: {sum(reward_history) / max_episode}')


if __name__ == '__main__':
    # configure the simulation
    number_of_simulation = 10
    env_discrete_version_set = [1, 2, 3, 4, 5, 6, 7, 8]  # check discrete_cartpole_env.py and discrete_pendulum_env.py for the version number
    max_episode = 500
    max_step = 250000  # SB3 agent uses max_step to terminate the training. Change this value to match the max_episode
    agent_name_set = ['DT', 'random'] # ['DT', 'random', 'DQN', 'PPO', 'A2C']  # choose agent from ['DT', 'random', 'DQN', 'PPO', 'A2C']
    env_name = 'CustomCartPole'  # choose environment from 'CustomCartPole', 'CustomPendulum'

    # single run test
    run_one_simulation('PPO', 0, env_name, max_episode, max_step)

    # for agent_name in agent_name_set:
    #     for env_discrete_version in env_discrete_version_set:
    #         print("run simulation: ", agent_name, env_discrete_version)
    #         # run the simulation in parallel, change the number of processes to match the number of CPU cores
    #         pool = mp.Pool(mp.cpu_count())
    #         for i in range(number_of_simulation):
    #             time.sleep(2)
    #             print("run simulation: ", i)
    #             pool.apply_async(run_one_simulation, args=(agent_name, env_discrete_version, env_name, max_episode, max_step))
    #         pool.close()
    #         pool.join()
    #         print("simulation finished: ", agent_name, env_discrete_version)

