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

env = gym.make("Pendulum")
print("sample action: ", env.action_space.sample(), type(env.action_space.sample()))
