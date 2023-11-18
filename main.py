'''
Project     : DT-DRL 
File        : main.py
Author      : Zelin Wan
Date        : 11/2/23
Description : main function to run all simulations.
'''
import gymnasium as gym
import datetime
import numpy as np
from discrete_cartpole_env import CustomCartPoleEnv
from discrete_pendulum_env import CustomPendulumEnv
from decision_theory_agent import DecisionTheoryCartpoleAgent
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback


def writer_log_scalar(writer, accumulated_reward, action_count_per_episode, episode, agent_name):
    writer.add_scalar('accumulated_reward/agent: {}'.format(agent_name), accumulated_reward, episode)
    action_ratio = action_count_per_episode / sum(action_count_per_episode)
    for i in range(len(action_ratio)):
        writer.add_scalar('action_ratio/agent: {}, action: {}'.format(agent_name, i), action_ratio[i], episode)
        writer.add_scalar('action_count/agent: {}, action: {}'.format(agent_name, i), action_count_per_episode[i],
                          episode)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        self.accumulated_reward = 0
        self.episode_count = 1
        self.action_count_per_episode = np.zeros(env.action_space.n)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        :return:
        """
        self.accumulated_reward += self.locals['rewards'][0]
        self.action_count_per_episode[self.locals['actions'][0]] += 1

        if self.locals.get('done') != False:
            # save to tensorboard
            writer_log_scalar(writer, self.accumulated_reward, self.action_count_per_episode, self.episode_count, agent_name)
            self.accumulated_reward = 0
            self.episode_count += 1
        return True


if __name__ == '__main__':
    # env = gym.make('CartPole-v1', render_mode='human')
    # env = gym.make('CartPole-v1'); env.env_name = 'OrignialCartPole'; env.discrete_version = 0
    env = CustomCartPoleEnv()
    # env = CustomPendulumEnv()


    # configure the simulation
    max_episode = 100
    max_step = 100000 # SB3 agent uses max_step to terminate the training. Change this value to match the max_episode
    agent_name = 'A2C'  # choose agent from ['DT', 'random', 'DQN', 'PPO', 'A2C']
    data_path = './data/' + env.env_name

    # create tensorboard writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer_path = data_path + '/agent_' + agent_name + '-discete_ver_' + str(env.discrete_version) + '-time_' + current_time
    writer = SummaryWriter(writer_path)


    # create agents and models
    DT_agent = None
    DQN_model = None
    PPO_model = None
    A2C_model = None
    if agent_name == 'DT':
        DT_agent = DecisionTheoryCartpoleAgent(env)
    elif agent_name == 'DQN':
        DQN_model = DQN('MlpPolicy', env, verbose=1)
        DQN_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback())
        exit(0)
    elif agent_name == 'PPO':
        PPO_model = PPO('MlpPolicy', env, verbose=1)
        PPO_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback())
        exit(0)
    elif agent_name == 'A2C':
        A2C_model = PPO('MlpPolicy', env, verbose=1)
        A2C_model.learn(total_timesteps=max_step, log_interval=1, callback=TensorboardCallback())
        exit(0)
    else:
        pass


    # run the simulation
    reward_history = []
    for episode in range(max_episode):
        obs, info = env.reset()  # fix seed for reproducibility
        terminated = False
        truncated = False
        accumulated_reward = 0
        action_count_per_episode = np.zeros(env.action_space.n)
        while not terminated or truncated:

            if agent_name == 'DT':
                action = DT_agent.get_action(obs)
            elif agent_name == 'DQN':
                # print("obs: ", obs)
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
                DT_agent.update_observation(obs, action, new_obs)

            # update the observation
            obs = new_obs

        print(f'Episode {episode} ended with accumulated reward {accumulated_reward}')
        # save to tensorboard
        writer.add_scalar('accumulated_reward/agent: {}'.format(agent_name), accumulated_reward, episode)
        action_ratio = action_count_per_episode / sum(action_count_per_episode)
        for i in range(len(action_ratio)):
            writer.add_scalar('action_ratio/agent: {}, action: {}'.format(agent_name, i), action_ratio[i], episode)
            writer.add_scalar('action_count/agent: {}, action: {}'.format(agent_name, i), action_count_per_episode[i],
                              episode)

        reward_history.append(accumulated_reward)

    print(f'Average reward over {max_episode} episodes: {sum(reward_history) / max_episode}')



