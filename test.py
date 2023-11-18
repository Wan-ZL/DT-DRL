'''
Project     : DT-DRL 
File        : test.py
Author      : Zelin Wan
Date        : 11/14/23
Description : 
'''
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

env = gym.make("CartPole-v1")

model = DQN("MlpPolicy", env, verbose=1, batch_size=3)
# model.learn(total_timesteps=100000, log_interval=4)
# model.save("dqn_cartpole")
#
# del model # remove to demonstrate saving and loading
#
# model = DQN.load("dqn_cartpole")

obs, info = env.reset()
accumulated_reward = 0
episodes = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    new_obs, reward, terminated, truncated, info = env.step(action)
    accumulated_reward += reward

    print('replay_buffer size: ', model.replay_buffer.size())
    print(obs, new_obs, action, reward, terminated, info)
    model.replay_buffer.add(obs=obs, next_obs=new_obs, action=action, reward=np.array(reward), done=np.array(terminated), infos=[info])
    print('replay_buffer size: ', model.replay_buffer.size(), 'replay_buffer contents: ', model.replay_buffer.observations, model.replay_buffer.next_observations, model.replay_buffer.actions, model.replay_buffer)
    model.train(0.001)

    obs = new_obs

    if terminated or truncated:
        obs, info = env.reset()
        print("accumulated_reward: ", accumulated_reward, "episodes: ", episodes)
        accumulated_reward = 0
        episodes += 1
