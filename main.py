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
import numpy as np
import multiprocessing as mp
from datetime import datetime

import torch

from DT_PPO import DecisionTheoryGuidedPPOAgent
from DT_PPO_3 import DecisionTheoryCombinedPPOAgent
from BFS_PPO import BFSGuidedPPOAgent
from discrete_cartpole_env import CustomCartPoleEnv
from discrete_pendulum_env import CustomPendulumEnv
from decision_theory_agent import DecisionTheoryAgent
from maze_decision_theory_agent import MazeDecisionTheoryAgent
from maze_BFS_agent import MazeBFSAgent
from maze_env import CustomMaze
from FPER_PPO import FPERPPOAgent
from gym_maze_2.envs.maze_env import MazeEnv
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from continuous_cartpole_old import ContinuousCartPoleEnv
from slightly_modified_cartpole import SlightlyModifiedCartPoleEnv



def writer_log_scalar(writer, accumulated_reward, action_count_per_episode, step_count, episode_count, agent_name):
    writer.add_scalar('accumulated_reward/agent: {}'.format(agent_name), accumulated_reward, episode_count)

    writer.add_scalar('step_count/agent: {}'.format(agent_name), step_count, episode_count)

    if action_count_per_episode is not None:
        action_ratio = action_count_per_episode / sum(action_count_per_episode)
        for i in range(len(action_ratio)):
            writer.add_scalar('action_ratio/agent: {}, action: {}'.format(agent_name, i), action_ratio[i], episode_count)
            writer.add_scalar('action_count/agent: {}, action: {}'.format(agent_name, i), action_count_per_episode[i],
                              episode_count)
    else:
        # for continuous action space, record the action 0 (do nothing)
        writer.add_scalar('action_ratio/agent: {}, action: {}'.format(agent_name, 0), 0, episode_count)
        writer.add_scalar('action_count/agent: {}, action: {}'.format(agent_name, 0), 0, episode_count)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting values in tensorboard.
    """

    def __init__(self, env, writer, agent_name, max_episode, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.writer = writer
        self.agent_name = agent_name
        # Those variables will be accessible in the callback
        self.accumulated_reward = 0
        self.step_count = 0
        self.episode_count = 1
        self.max_episode = max_episode
        self.action_count_per_episode = np.zeros(self.env.action_space.n) if type(
            self.env.action_space) == gym.spaces.discrete.Discrete else None

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return:
        """
        self.accumulated_reward += self.locals['rewards'][0]
        self.step_count += 1
        if self.action_count_per_episode is not None:
            self.action_count_per_episode[self.locals['actions'][0]] += 1

        # check if the episode is done
        episode_done = False
        # check if algorithm is PPO or A2C
        class_name_list = ['PPO', 'A2C', 'DecisionTheoryGuidedPPOAgent']
        if self.model.__class__.__name__ in class_name_list:
            episode_done = self.locals.get('done')
        elif self.model.__class__.__name__ == 'DQN':
            episode_done = self.locals.get('dones')[0]


        if episode_done:
            # episode ended, save to tensorboard
            writer_log_scalar(self.writer, self.accumulated_reward, self.action_count_per_episode, self.step_count, self.episode_count,
                              self.agent_name)
            self.accumulated_reward = 0
            self.step_count = 0
            self.episode_count += 1
            self.action_count_per_episode = np.zeros(self.env.action_space.n) if type(
                self.env.action_space) == gym.spaces.discrete.Discrete else None

        if self.episode_count >= self.max_episode:
            return False
        else:
            return True


def run_simulation(agent_name, env_discrete_version, env_name, max_episode, max_step, fix_seed, transfer_time_point=0):
    # env = gym.make('CartPole-v1', render_mode='human')
    # env = gym.make('CartPole-v1'); env.env_name = 'OrignialCartPole'; env.discrete_version = 0
    if env_name == 'CustomCartPole':
        env = CustomCartPoleEnv(env_discrete_version, fix_seed)
    elif env_name == 'SlightlyModifiedCartPole':
        env = SlightlyModifiedCartPoleEnv(env_discrete_version, fix_seed)
    elif env_name == 'CustomPendulum':
        env = CustomPendulumEnv(env_discrete_version, fix_seed)
    elif env_name == 'CustomMaze':
        env = CustomMaze(env_discrete_version, fix_seed, enable_render=False)
        # env = CustomMaze(maze_size=(5, 5), fix_seed=fix_seed, enable_render=False)
        # env = CustomMaze(maze_file="maze2d_5x5.npy", enable_render=True)
    else:
        raise Exception('Invalid environment name')
    new_env = env   # use new_env in TL_PPO

    # create tensorboard writer
    if fix_seed is None:
        path_seed_name = 'RandomSeed'
    else:
        path_seed_name = 'FixSeed'


    data_path = './data/' + path_seed_name + '/' + env.env_name + '/agent_' + agent_name + '/env_discrete_ver_' + str(env.discrete_version)
    if transfer_time_point != 0:
        data_path += '/transfer_time_point_' + str(transfer_time_point)


    # create folder if not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # time stamp (year, month, day, hour, minute, second, millisecond)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer_path = data_path + '/agent_' + agent_name + '-env_discrete_ver_' + str(
        env.discrete_version) + '-time_' + current_time
    writer = SummaryWriter(writer_path)

    # create agents and models
    DT_agent = None
    BFS_agent = None
    DQN_model = None
    PPO_model = None
    PPO_model_2 = None
    A2C_model = None
    DT_PPO_model = None
    BFS_PPO_model = None
    expert_PPO_model = None
    if agent_name == 'DT':
        if env_name == 'CustomMaze':
            DT_agent = MazeDecisionTheoryAgent(env)
        else:
            DT_agent = DecisionTheoryAgent(env)
    elif agent_name == 'BFS':
        if env_name == 'CustomMaze':
            BFS_agent = MazeBFSAgent(env)
        else:
            raise Exception('BFS agent does not support this environment')
    elif agent_name == 'DQN':
        DQN_model = DQN('MlpPolicy', env, verbose=1)
        DQN_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback(env, writer, agent_name, max_episode))
        return
    elif agent_name == 'PPO':
        PPO_model = PPO('MlpPolicy', env, verbose=1)
        PPO_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback(env, writer, agent_name, max_episode))
        return
    elif agent_name == 'PPO_2':
        env.num_envs = 1
        PPO_model_2 = PPO('MlpPolicy', env, verbose=1)
        PPO_model_2._setup_learn(
            total_timesteps=max_step,
            callback=TensorboardCallback(env, writer, agent_name, max_episode),
            reset_num_timesteps=True,
            tb_log_name="OnPolicyAlgorithm",
            progress_bar=False,
        )
        n_steps = 0
    elif agent_name == 'TL_PPO':
        # transfer learning with PPO (first train on maze=3x3, then transfer to other maze). TL_PPO only designed for CustomMaze environment.
        if env_name != 'CustomMaze':
            raise Exception('TL_PPO only support CustomMaze environment')

        env.num_envs = 1
        PPO_model_2 = PPO('MlpPolicy', env, verbose=1)
        PPO_model_2._setup_learn(
            total_timesteps=max_step,
            callback=TensorboardCallback(env, writer, agent_name, max_episode),
            reset_num_timesteps=True,
            tb_log_name="OnPolicyAlgorithm",
            progress_bar=False,
        )
        n_steps = 0
        # use old_env to train the PPO model for first 100 episodes, then switch to new_env
        old_env = CustomMaze(3, fix_seed, enable_render=False)
        old_env.num_envs = 1
        env = old_env
    elif agent_name == 'FPER_PPO':
        env.num_envs = 1
        PPO_model_2 = FPERPPOAgent('MlpPolicy', env, verbose=1, n_epochs=20)
        PPO_model_2._setup_learn(
            total_timesteps=max_step,
            callback=TensorboardCallback(env, writer, agent_name, max_episode),
            reset_num_timesteps=True,
            tb_log_name="OnPolicyAlgorithm",
            progress_bar=False,
        )
        n_steps = 0
    elif agent_name == 'IL_PPO':
        # Imiation learning with PPO. IL_PPO only designed for Cart Pole environment.
        env_for_expert = SlightlyModifiedCartPoleEnv(env_discrete_version, fix_seed)
        expert_PPO_model = PPO('MlpPolicy', env_for_expert, verbose=1) # This is expert model will be used for imitation learning
        expert_PPO_model.learn(total_timesteps=10000)  # train the expert model for 10000 steps.
        imitation_episodes = 300 # imitate learn the expert model for 100 episodes
        use_expert = True
        env.num_envs = 1
        PPO_model_2 = PPO('MlpPolicy', env, verbose=1)
        PPO_model_2._setup_learn(
            total_timesteps=max_step,
            callback=TensorboardCallback(env, writer, agent_name, max_episode),
            reset_num_timesteps=True,
            tb_log_name="OnPolicyAlgorithm",
            progress_bar=False,
        )
        n_steps = 0
    elif agent_name == 'A2C':
        A2C_model = A2C('MlpPolicy', env, verbose=1)
        A2C_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback(env, writer, agent_name, max_episode))
        return
    elif agent_name == 'DT_PPO':    # use the action distribution of DT to initialize the PPO model
        DT_PPO_model = DecisionTheoryGuidedPPOAgent('MlpPolicy', env, fix_seed=fix_seed, verbose=1)    # learning_rate=0.000000 for testing
        DT_PPO_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback(env, writer, agent_name, max_episode))
        return
    elif agent_name == 'DT_PPO_2':  # use DT first to make decision (train PPO in background), then switch to PPO after certain episodes
        if env_name == 'CustomMaze':
            DT_agent = MazeDecisionTheoryAgent(env)
            # DT_agent = MazeBFSAgent(env)
        else:
            DT_agent = DecisionTheoryAgent(env)
        use_DT = True
        env.num_envs = 1
        PPO_model_2 = PPO('MlpPolicy', env, verbose=1)
        PPO_model_2._setup_learn(
            total_timesteps=max_step,
            callback=TensorboardCallback(env, writer, agent_name, max_episode),
            reset_num_timesteps=True,
            tb_log_name="OnPolicyAlgorithm",
            progress_bar=False,
        )
        n_steps = 0
    elif agent_name == 'DT_PPO_3':  # use utilities of DT to combine with the output of PPO action network. Reduce the weight of utility gradually.
        if env_name == 'CustomMaze':
            DT_agent = MazeDecisionTheoryAgent(env)
        else:
            DT_agent = DecisionTheoryAgent(env)
        env.num_envs = 1
        PPO_model_3 = DecisionTheoryCombinedPPOAgent('MlpPolicy', env, DT_agent=DT_agent, verbose=1)
        PPO_model_3._setup_learn(
            total_timesteps=max_step,
            callback=TensorboardCallback(env, writer, agent_name, max_episode),
            reset_num_timesteps=True,
            tb_log_name="OnPolicyAlgorithm",
            progress_bar=False,
        )
        n_steps = 0
    elif agent_name == 'BFS_PPO':
        BFS_PPO_model = BFSGuidedPPOAgent('MlpPolicy', env, verbose=1)
        BFS_PPO_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback(env, writer, agent_name, max_episode))
        return
    else:
        pass

    # run the simulation
    reward_history = []
    last_100_action_ratio = np.zeros((1, env.action_space.n))
    buffer_full = False
    log_to_TensorBoard = False if agent_name == 'TL_PPO' or agent_name == 'IL_PPO' else True   # This is a hack to insure only TL_PPO switch to new_env once.
    episode = 0
    while episode < max_episode:
        if episode > 100 and agent_name == 'TL_PPO' and buffer_full and not log_to_TensorBoard:
            # use the new environment to train the PPO model after 100 episodes
            print("switch to new environment")
            env = new_env
            log_to_TensorBoard = True
            episode = 0 # reset the episode count for new environment

        obs, info = env.reset()  # fix seed for reproducibility
        if agent_name == 'DT':
            # DT agent use the discrete observation instead of mapped observation. dds
            if info.get('discrete_obs') is not None:
                obs = info['discrete_obs']
        elif agent_name == 'PPO_2' or agent_name == 'TL_PPO' or agent_name == 'FPER_PPO' or agent_name == 'IL_PPO':
            # add a dimension to the observation
            PPO_model_2._last_obs = np.expand_dims(obs, axis=0)

        terminated = False
        truncated = False
        accumulated_reward = 0
        step_counter = 0
        action_count_per_episode = np.zeros(env.action_space.n)
        while not terminated or not truncated:
            if agent_name == 'DT':
                action = DT_agent.get_action(obs)
            elif agent_name == 'BFS':
                action = BFS_agent.get_action(obs)
            elif agent_name == 'DQN':
                action, _states = DQN_model.predict(obs)
            elif agent_name == 'PPO':
                action, _states = PPO_model.predict(obs)
            elif agent_name == 'PPO_2' or agent_name == 'TL_PPO' or agent_name == 'FPER_PPO':
                if PPO_model_2.use_sde and PPO_model_2.sde_sample_freq > 0 and n_steps % PPO_model_2.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    PPO_model_2.policy.reset_noise(env.num_envs)
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(PPO_model_2._last_obs, PPO_model_2.device)
                    actions, values, log_probs = PPO_model_2.policy(obs_tensor)
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions
                # Clip the actions to avoid out of bound error
                if isinstance(PPO_model_2.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions, PPO_model_2.action_space.low, PPO_model_2.action_space.high)
                action = clipped_actions
            elif agent_name == 'IL_PPO':
                if episode >= imitation_episodes and use_expert:
                    use_expert = False
                    log_to_TensorBoard = True
                    episode = 0  # reset the episode count for new environment

                # expert model make decision
                if use_expert:
                    expert_action, _states = expert_PPO_model.predict(PPO_model_2._last_obs)

                if PPO_model_2.use_sde and PPO_model_2.sde_sample_freq > 0 and n_steps % PPO_model_2.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    PPO_model_2.policy.reset_noise(env.num_envs)
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(PPO_model_2._last_obs, PPO_model_2.device)
                    PPO_actions, values, log_probs = PPO_model_2.policy(obs_tensor)
                PPO_actions = PPO_actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = PPO_actions
                # Clip the actions to avoid out of bound error
                if isinstance(PPO_model_2.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(PPO_actions, PPO_model_2.action_space.low, PPO_model_2.action_space.high)
                PPO_actions = clipped_actions
                if use_expert:
                    action = expert_action
                    actions = expert_action
                else:
                    action = PPO_actions
                    actions = PPO_actions

            elif agent_name == 'A2C':
                action, _states = A2C_model.predict(obs)
            elif agent_name == 'DT_PPO':
                action, _states = DT_PPO_model.predict(obs)
            elif agent_name == 'DT_PPO_2':
                # DT agent make decision
                DT_action = DT_agent.get_action(obs)

                # PPO agent make decision
                if PPO_model_2.use_sde and PPO_model_2.sde_sample_freq > 0 and n_steps % PPO_model_2.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    PPO_model_2.policy.reset_noise(env.num_envs)
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(PPO_model_2._last_obs, PPO_model_2.device)
                    PPO_actions, values, log_probs = PPO_model_2.policy(obs_tensor)
                PPO_actions = PPO_actions.cpu().numpy()
                # Rescale and perform action
                clipped_actions = PPO_actions
                # Clip the actions to avoid out of bound error
                if isinstance(PPO_model_2.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(PPO_actions, PPO_model_2.action_space.low, PPO_model_2.action_space.high)
                PPO_actions = clipped_actions

                if use_DT:
                    if episode > transfer_time_point:  # switch to PPO after this episodes
                        use_DT = False
                    action = DT_action
                    actions = np.array([DT_action])
                else:
                    action = PPO_actions
                    actions = PPO_actions
            elif agent_name == 'DT_PPO_3':
                # PPO agent make decision
                if PPO_model_3.use_sde and PPO_model_3.sde_sample_freq > 0 and n_steps % PPO_model_3.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    PPO_model_3.policy.reset_noise(env.num_envs)
                with torch.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(PPO_model_3._last_obs, PPO_model_3.device)
                    PPO_actions, values, log_probs = PPO_model_3.policy(obs_tensor)
                PPO_actions = PPO_actions.cpu().numpy()
                # Rescale and perform action
                clipped_actions = PPO_actions
                # Clip the actions to avoid out of bound error
                if isinstance(PPO_model_3.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(PPO_actions, PPO_model_3.action_space.low, PPO_model_3.action_space.high)
                PPO_actions = clipped_actions

                action = PPO_actions
                actions = PPO_actions

            elif agent_name == 'BFS_PPO':
                action, _states = BFS_PPO_model.predict(obs)
            else:
                action = env.action_space.sample()
            # count the action
            action_count_per_episode[action] += 1

            # print("action: ", action, type(action))
            new_obs, reward, terminated, truncated, info = env.step(action)
            # print("new_obs: ", new_obs, type(new_obs), "reward: ", reward, type(reward), "terminated: ", terminated, type(terminated), "truncated: ", truncated, type(truncated), "info: ", info, type(info))
            if agent_name == 'DT' or agent_name == 'BFS' or agent_name == 'DT_PPO_2' or agent_name == 'DT_PPO_3':
                # DT agent use the discrete observation instead of mapped observation.
                if info.get('discrete_obs') is not None:
                    new_obs = info['discrete_obs']

            accumulated_reward += reward
            step_counter += 1
            # if accumulated_reward >= 100:
            #     env.render_mode = 'human'

            # Observe the environment or update the belief
            if agent_name == 'DT':
                DT_agent.update_observation(obs, action, new_obs, reward)
            elif agent_name == 'BFS':
                BFS_agent.update_observation(obs, action, new_obs, reward)
            elif agent_name == 'PPO_2' or agent_name == 'DT_PPO_2' or agent_name == 'TL_PPO' or agent_name == 'FPER_PPO' or agent_name == 'IL_PPO':
                if agent_name == 'DT_PPO_2':
                    DT_agent.update_observation(obs, action, new_obs, reward)

                PPO_model_2.num_timesteps += env.num_envs

                # PPO_model_2._update_info_buffer(truncated, info)
                n_steps += 1

                if isinstance(PPO_model_2.action_space, gym.spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                PPO_model_2.rollout_buffer.add(
                    PPO_model_2._last_obs,
                    actions, reward,
                    PPO_model_2._last_episode_starts,
                    values, log_probs)
                PPO_model_2._last_obs = np.expand_dims(new_obs, axis=0)
                PPO_model_2._last_episode_starts = np.expand_dims(terminated, axis=0)

                if PPO_model_2.rollout_buffer.full:
                    if episode > 100:   # This is a hack to enable the new env after 100 episodes when memory buffer is full
                        buffer_full = True

                    PPO_model_2.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=terminated)
                    print("train PPO here")
                    print("train episode", PPO_model_2.n_epochs)
                    if 'use_DT' in locals():
                        print("use_DT: ", use_DT)
                    PPO_model_2.policy.set_training_mode(True)
                    PPO_model_2.train()  # For training the model
                    PPO_model_2.policy.set_training_mode(False)
                    PPO_model_2.rollout_buffer.reset()
                    n_steps = 0
            elif agent_name == 'DT_PPO_3':
                PPO_model_3.policy.DT_agent.update_observation(obs, action, new_obs, reward) if PPO_model_3.policy.DT_agent is not None else None

                PPO_model_3.num_timesteps += env.num_envs

                n_steps += 1

                if isinstance(PPO_model_3.action_space, gym.spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)
                PPO_model_3.rollout_buffer.add(
                    PPO_model_3._last_obs,
                    actions, reward,
                    PPO_model_3._last_episode_starts,
                    values, log_probs)
                PPO_model_3._last_obs = np.expand_dims(new_obs, axis=0)
                PPO_model_3._last_episode_starts = np.expand_dims(terminated, axis=0)

                if PPO_model_3.rollout_buffer.full:
                    PPO_model_3.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=terminated)
                    print("train PPO here")
                    print("train episode", PPO_model_3.n_epochs)
                    if 'use_DT' in locals():
                        print("use_DT: ", use_DT)
                    PPO_model_3.policy.set_training_mode(True)
                    PPO_model_3.train()  # For training the model
                    PPO_model_3.policy.set_training_mode(False)
                    PPO_model_3.rollout_buffer.reset()
                    n_steps = 0



            # update the observation
            obs = new_obs

            if terminated or truncated:
                break

        print(f'Episode {episode} ended with accumulated reward {accumulated_reward}')

        if not log_to_TensorBoard:
            print("Not log to TensorBoard")
            episode += 1
            continue    # do not save the data for the old environment

        # save to tensorboard
        writer.add_scalar('accumulated_reward/agent: {}'.format(agent_name), accumulated_reward, episode)
        writer.add_scalar('step_count/agent: {}'.format(agent_name), step_counter, episode)
        action_ratio = action_count_per_episode / sum(action_count_per_episode)
        # record the last 100 action ratio
        if len(last_100_action_ratio) >= 100:
            last_100_action_ratio = np.delete(last_100_action_ratio, 0, 0)
        last_100_action_ratio = np.append(last_100_action_ratio, [action_ratio], axis=0)


        for i in range(len(action_ratio)):
            writer.add_scalar('action_ratio/agent: {}, action: {}'.format(agent_name, i), action_ratio[i], episode)
            writer.add_scalar('action_count/agent: {}, action: {}'.format(agent_name, i),
                              action_count_per_episode[i], episode)

        if agent_name == 'DT_PPO_3':
            writer.add_scalar('others/u_weight in DT_PPO_3', PPO_model_3.policy.u_weight, episode)

        reward_history.append(accumulated_reward)
        episode += 1

    print(f'Average reward over {max_episode} episodes: {sum(reward_history) / max_episode}')

    # get the mean of the last 100 action ratio
    last_100_action_ratio += 0.01 # add a small number to avoid zero for each action
    last_100_action_ratio_mean = np.mean(last_100_action_ratio, axis=0)
    # write the last 100 action ratio to file. Allow multiple processes to write to the same file
    write_path_for_action_ratio = data_path + '/last_100_action_ratio'
    if not os.path.exists(write_path_for_action_ratio):
        os.makedirs(write_path_for_action_ratio)
    np.save(write_path_for_action_ratio + '/last_100_action_ratio_list-' + current_time + '.npy', last_100_action_ratio_mean)

def tb_reducer_mean_calculation(env_discrete_version_set, agent_name_set, transfer_time_point_set, env_name, fix_seed):
    # run bash script tb_reducer_script.sh to generate the data. Run tb_reducer_script.sh in parallel to speed up the process
    # create tensorboard writer
    if fix_seed is None:
        seed_path_name = 'RandomSeed'
    else:
        seed_path_name = 'FixSeed'

    cpu_count = mp.cpu_count()
    path = os.getcwd() + '/data/' + seed_path_name + '/'
    print("path: ", path)
    for env_discrete_version in env_discrete_version_set:
        for agent_name in agent_name_set:
            if agent_name != 'DT_PPO_2':
                read_path = path + env_name + '/agent_' + agent_name + '/env_discrete_ver_' + str(env_discrete_version) + '/*'
                write_path = path + 'tb_reduce/' + env_name + '/agent_' + agent_name + '/env_discrete_ver_' + str(env_discrete_version) + '/'
                print("read_path: ", read_path)
                print("write_path: ", write_path)
                os.system('tb-reducer {} -o {} -r mean --handle-dup-steps \'mean\' --lax-step --lax-tags'.format(read_path, write_path))
            else:
                for transfer_time_point in transfer_time_point_set:
                    read_path = path + env_name + '/agent_' + agent_name + '/env_discrete_ver_' + str(env_discrete_version) + '/transfer_time_point_' + str(transfer_time_point) + '/*'
                    write_path = path + 'tb_reduce/' + env_name + '/agent_' + agent_name + '/env_discrete_ver_' + str(env_discrete_version) + '/transfer_time_point_' + str(transfer_time_point) + '/'
                    print("read_path: ", read_path)
                    print("write_path: ", write_path)
                    os.system('tb-reducer {} -o {} -r mean --handle-dup-steps \'mean\' --lax-step --lax-tags'.format(read_path, write_path))




if __name__ == '__main__':
    # configure the simulation
    number_of_simulation = 100
    env_discrete_version_set = [1] #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] #[1, 2, 3, 4, 5, 6, 7, 8, 9]  # check discrete_cartpole_env_old.py and discrete_pendulum_env.py for the version number
    max_episode = 500 #1000
    max_step = 2250000 #250000  # if max_episode is reached before max_step, the simulation will stop.
    agent_name_set = ['IL_PPO'] #['DT', 'DQN', 'DT_PPO_3', 'PPO_2']  # choose agents from ['DT', 'BFS', 'random', 'DQN', 'PPO', 'PPO_2', 'A2C', 'DT_PPO', 'DT_PPO_2', 'DT_PPO_3', 'BFS_PPO', 'TL_PPO', 'FPER_PPO', 'IL_PPO']
    transfer_time_point_set = [100] # the time point to transfer from DT to PPO [10, 30, 50, 70, 100, 200, 300, 400, 500, 600]
    env_name = 'SlightlyModifiedCartPole'  # choose environment from 'SlightlyModifiedCartPole', 'CustomCartPole', 'CustomPendulum', 'CustomMaze'
    fix_seeds = [np.random.randint(0, 100000) for i in range(number_of_simulation)]
    fix_seed = 123 # None means random seed, otherwise fix the seed to a specific number.

    single_test = False # set to True to run a single simulation
    if single_test:
        # single run test
        test_agent = 'IL_PPO'
        env_discrete_version = 1
        if test_agent == 'DT_PPO_2':
            transfer_time_point = 10
        else:
            transfer_time_point = 0
        start_time = time.time()
        run_simulation(test_agent, env_discrete_version, env_name, max_episode, max_step, fix_seed, transfer_time_point)
        # save running time (in second) to file
        running_time = time.time() - start_time
        if fix_seed is None:
            path_seed_name = 'RandomSeed'
        else:
            path_seed_name = 'FixSeed'
        data_path = './data/' + path_seed_name + '/' + env_name + '/agent_' + test_agent + '/env_discrete_ver_' + str(env_discrete_version)
        with open(data_path + '/running_time.txt', 'w') as f:
            f.write(str(running_time))
    else:
        for agent_name in agent_name_set:
            for env_discrete_version in env_discrete_version_set:
                print("run simulation: ", agent_name, env_discrete_version, 'seeds:', fix_seeds)
                if agent_name != 'DT_PPO_2':
                    _transfer_time_point_set = [0]
                else:
                    _transfer_time_point_set = transfer_time_point_set

                for transfer_time_point in _transfer_time_point_set:
                    # run the simulation in parallel, change the number of processes to match the number of CPU cores
                    pool = mp.Pool(mp.cpu_count())
                    start_time = time.time()
                    for i in range(number_of_simulation):
                        time.sleep(2)
                        print("run simulation: ", i)
                        pool.apply_async(run_simulation, args=(agent_name, env_discrete_version, env_name, max_episode, max_step, fix_seeds[i], transfer_time_point))
                    pool.close()
                    pool.join()

                    print("simulation finished: ", agent_name, env_discrete_version)

                    # save running time (in second) to file
                    running_time = time.time() - start_time
                    if fix_seed is None:
                        path_seed_name = 'RandomSeed'
                    else:
                        path_seed_name = 'FixSeed'
                    data_path = './data/' + path_seed_name + '/' + env_name + '/agent_' + agent_name + '/env_discrete_ver_' + str(env_discrete_version)
                    with open(data_path + '/running_time.txt', 'w') as f:
                        f.write(str(running_time))

        tb_reducer_mean_calculation(env_discrete_version_set, agent_name_set, transfer_time_point_set, env_name, fix_seeds)



