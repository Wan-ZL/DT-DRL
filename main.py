'''
Project     : DT-DRL 
File        : main.py
Author      : Zelin Wan
Date        : 11/2/23
Description : main function to run all simulations.
'''
import gym
import datetime

import numpy as np

from discrete_cartpole_env import CustomCartPoleEnv
from discrete_pendulum_env import CustomPendulumEnv
from decision_theory_agent import DecisionTheoryCartpoleAgent
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    # env = gym.make('CartPole-v1', render_mode='human')
    # env = CustomCartPoleEnv()
    env = CustomPendulumEnv()


    # configure the simulation
    max_episode = 1000
    agent_name = 'random'  # choose agent from ['DT', 'random']
    data_path = './data/' + env.env_name

    # create tensorboard writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(data_path + '/agent_' + agent_name + '-discete_ver_' + str(env.discrete_version) +
                           '-time_' + current_time)

    # create the decision theory agent
    DT_agent = DecisionTheoryCartpoleAgent(env)

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
            else:
                action = env.action_space.sample()
            # count the action
            action_count_per_episode[action] += 1

            obs_, reward, terminated, truncated, info = env.step(action)
            accumulated_reward += reward

            # DT agent observe the environment and update the belief
            DT_agent.update_observation(obs, action, obs_)

            # update the observation
            obs = obs_

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
