'''
Project     ：DT-DRL 
File        ：main.py
Author      ：Zelin Wan
Date        ：11/2/23
Description : main function to run all simulations.
'''
import gym

from discret_env import CustomCartPoleEnv
from decision_theory_agent import DecisionTheoryCartpoleAgent





if __name__ == '__main__':
    # env = gym.make('CartPole-v1', render_mode='human')
    env = CustomCartPoleEnv()

    # create the decision theory agent
    DT_agent = DecisionTheoryCartpoleAgent(env)

    # run the simulation
    reward_history = []
    max_episode = 1000
    for episode in range(max_episode):
        obs, info = env.reset()   # fix seed for reproducibility
        terminated = False
        truncated = False
        accumulated_reward = 0
        while not terminated or truncated:
            # action = env.action_space.sample()
            action = DT_agent.get_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            accumulated_reward += reward


            # DT agent observe the environment and update the belief
            DT_agent.update_observation(obs, action, obs_)

            # update the observation
            obs = obs_


        print(f'Episode {episode} ended with accumulated reward {accumulated_reward}')
        reward_history.append(accumulated_reward)

    print(f'Average reward over {max_episode} episodes: {sum(reward_history) / max_episode}')





