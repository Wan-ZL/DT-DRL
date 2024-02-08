'''
Project     : DT-DRL 
File        : DT_PPO.py
Author      : Zelin Wan
Date        : 1/16/24
Description : This PPO try to use the output of action network multi the utility of decision theory to get the final action.
'''

from stable_baselines3.ppo.ppo import *
from stable_baselines3.common.policies import ActorCriticCnnPolicy, BasePolicy, MultiInputActorCriticPolicy
from DT_policies import CustomActorCriticPolicy as ActorCriticPolicy
from maze_decision_theory_agent import MazeDecisionTheoryAgent
from decision_theory_agent import DecisionTheoryAgent

class DecisionTheoryCombinedPPOAgent(PPO):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }
    def __init__(self, policy, env, DT_agent=None, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.policy.DT_agent = DT_agent
        # self.policy.DT_agent = MazeDecisionTheoryAgent(env)
        # self.policy.DT_agent = DecisionTheoryAgent(env)




