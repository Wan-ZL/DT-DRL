'''
Project     : DT-DRL 
File        : maze_BFS_agent.py
Author      : Zelin Wan
Date        : 12/21/23
Description : 
'''

from copy import deepcopy
from time import sleep

import numpy as np
from collections import defaultdict
from queue import Queue


class MazeBFSAgent:
    def __init__(self, env):
        self.env = env
        self.start_state = env.start_state
        self.goal_state = env.goal_state

        # True means the state is transitionable, False means the state is not transitionable (a wall)
        self.state_transitionable_dict = defaultdict(lambda: True)  # input is old_state + action in tuple format.

        self.old_state = (0, 0)

        self.g_value_dict = defaultdict(lambda: float('inf'))  # initialize g_value to infinity for all states
        self.g_value_dict[self.start_state] = 0  # g_value of start state is 0
        self.shotest_distance_from_origin = defaultdict(lambda: float('inf'))  # initialize the shortest distance from the origin to infinity for all states
        self.shotest_distance_from_origin[(0, 0)] = 0  # the shortest distance from the origin to the origin is 0
        self.previous_state_dict = defaultdict(lambda: None)  # initialize the previous state to None for all states
        self.previous_state_dict[(0, 0)] = (0, 0)  # the previous state of the origin is the origin itself
        self.state_visited = defaultdict(lambda: False)  # initialize the state note visited to True for all states
        self.action_list = []   # a list of actions (first in first out)
        self.visited = set()
        self.unvisited_states = set()  # list to store unvisited neighbor states
        self.unvisited_states_pre = set()  # list to store the state where its neighbor has unvisited states

        self.visited_map = defaultdict(lambda: False)  # initialize the visited map to False for all states
        self.visited_map[self.start_state] = True  # the start state is visited
        self.queue = Queue()    # BFS queue. State and an action list to that state are stored in the queue
        self.queue.put((self.start_state, []))
        self.path_from_origin = defaultdict(lambda: [])  # list to save the path from the origin to each state
        self.path_from_origin[self.start_state] = []  # the path from the origin to the origin is empty list
        self.explore_list = []  # a list of states to explore
        self.final_action_list_to_goal = None  # a list of actions to the goal state

    def calculate_manhattan_distance(self, state1, state2):
        return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

    def calculate_g_value(self, parent_state, current_state):
        g_value = self.g_value_dict[parent_state] + 1  # cost to move from one state to another is 1
        self.g_value_dict[current_state] = min(g_value, self.g_value_dict[current_state])
        print("g_value_dict: ", self.g_value_dict)
        return self.g_value_dict[current_state]



    def get_action_list_to_goal(self, current_state, goal_state):
        '''
        Get the action list to the goal state. First move from current_state to the (0, 0), then move from (0, 0) to the goal state.
        :param goal_state:
        :return: action list, None if the goal state cannot be reached from the current state
        '''
        if self.previous_state_dict[tuple(goal_state)] is None:
            return None

        action_list_from_current_state_to_origin = []
        action_list_from_origin_to_goal_state = []

        # move from current_state to the (0, 0)
        current_state = deepcopy(current_state)
        while current_state != (0, 0):
            print("finding action list to the origin, current_state: ", current_state)
            previous_state = self.previous_state_dict[tuple(current_state)]
            if previous_state[0] - current_state[0] == 1:
                action_list_from_current_state_to_origin.append(2)
            elif previous_state[0] - current_state[0] == -1:
                action_list_from_current_state_to_origin.append(3)
            elif previous_state[1] - current_state[1] == 1:
                action_list_from_current_state_to_origin.append(1)
            elif previous_state[1] - current_state[1] == -1:
                action_list_from_current_state_to_origin.append(0)
            else:
                raise Exception('Invalid action')
            current_state = previous_state

        # move from (0, 0) to the goal state. (use self.shotest_distance_from_origin but reverse the direction)
        goal_state = deepcopy(goal_state)
        while goal_state != (0, 0):
            print("finding action list to the goal, goal_state: ", goal_state)
            pre_goal_state = self.previous_state_dict[tuple(goal_state)]
            if pre_goal_state[0] - goal_state[0] == 1:
                action_list_from_origin_to_goal_state.append(3)
            elif pre_goal_state[0] - goal_state[0] == -1:
                action_list_from_origin_to_goal_state.append(2)
            elif pre_goal_state[1] - goal_state[1] == 1:
                action_list_from_origin_to_goal_state.append(0)
            elif pre_goal_state[1] - goal_state[1] == -1:
                action_list_from_origin_to_goal_state.append(1)
            else:
                raise Exception('Invalid action')
            goal_state = pre_goal_state
        # reverse the action list from (0, 0) to the goal state
        action_list_from_origin_to_goal_state.reverse()

        # combine the two action lists
        action_list = action_list_from_current_state_to_origin + action_list_from_origin_to_goal_state
        return action_list



    def get_new_state(self, state, action):
        new_state = list(state)
        if action == 0:  # up
            new_state[1] -= 1
        elif action == 1:  # down
            new_state[1] += 1
        elif action == 2:  # right
            new_state[0] += 1
        elif action == 3:  # left
            new_state[0] -= 1
        else:
            raise Exception('Invalid action')

        new_state = tuple(new_state)

        # check if the new state is a wall
        if not self.state_transitionable_dict[(state, action)]:
            return state

        # check if the new state is out of bound
        if not self.is_valid_state(new_state):
            return state

        return new_state


    def is_valid_state(self, state):
        x, y = state
        if x < 0 or x >= self.env.maze_size[0] or y < 0 or y >= self.env.maze_size[1]:
            return False  # state is out of maze boundaries
        if not self.state_transitionable_dict[state]:
            return False  # state is a wall
        return True



    def get_action(self, obs):
        # if self.final_action_list_to_goal is not None:
        #     sleep(0.05)
        # If there are actions left in the action_list, execute them first
        if self.action_list:
            return self.action_list.pop(0)

        # if the path to the goal state is found, return the action list to the goal state
        if self.final_action_list_to_goal is not None:
            self.action_list = deepcopy(self.final_action_list_to_goal)
            return self.action_list.pop(0)

        # If action_list is empty, find a new path using BFS
        # (node, path) = self.queue.get()
        # (node, path) = self.explore_list.pop(0)
        # add to explore list
        for action in range(self.env.action_space.n):
            next_state = self.get_new_state(tuple(obs), action)
            if next_state not in self.visited_map:
                self.explore_list.append((next_state, self.path_from_origin[tuple(obs)] + [action]))


        # for action in range(self.env.action_space.n):
        #     self.explore_list.append((self.get_new_state(node, action), path + [action]))
        (next_state, next_path) = self.explore_list.pop(0)
        # move to the next state. Move to start state first, then move to the next state
        self.action_list = self.get_path_from_current_to_start(tuple(obs)) + next_path
        return self.action_list.pop(0)

    def get_path_from_current_to_start(self, current_state):
        path_to_current = self.path_from_origin[current_state]
        # reverse the actions in the path
        reverse_path_to_current = []
        for index in range(len(path_to_current)):
            if path_to_current[index] == 0:
                reverse_path_to_current.append(1)
            elif path_to_current[index] == 1:
                reverse_path_to_current.append(0)
            elif path_to_current[index] == 2:
                reverse_path_to_current.append(3)
            elif path_to_current[index] == 3:
                reverse_path_to_current.append(2)
            else:
                raise Exception('Invalid action')
        reverse_path_to_current.reverse()
        return reverse_path_to_current


    def update_observation(self, obs, action, new_obs, reward):
        '''
        Update the observation
        :param obs: old state
        :param action: action
        :param new_obs: new state
        :param reward: reward
        :return:
        '''
        if type(action) == np.ndarray:
            action = action[0]
        if tuple(new_obs) == self.old_state:
            # sleep(0.5)
            self.state_transitionable_dict[(tuple(new_obs), action)] = False
        else:
            if not self.visited_map[tuple(new_obs)]:
                self.path_from_origin[tuple(new_obs)] = self.path_from_origin[self.old_state] + [action]
                self.visited_map[tuple(new_obs)] = True

        self.old_state = tuple(new_obs)

        # if the goal state is reached, save the action list (path) to the goal state
        if tuple(new_obs) == self.goal_state:
            self.final_action_list_to_goal = self.path_from_origin[self.goal_state]

