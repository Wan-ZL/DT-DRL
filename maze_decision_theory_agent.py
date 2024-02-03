'''
Project     : DT-DRL 
File        : maze_decision_theory_agent.py
Author      : Zelin Wan
Date        : 12/18/23
Description : The decision theory agent for the maze environment
'''
import random
from copy import deepcopy
from time import sleep

import numpy as np
from collections import defaultdict
from queue import Queue


class MazeDecisionTheoryAgent:
    def __init__(self, env):
        self.env = env
        self.start_state = env.start_state
        self.goal_state = env.goal_state
        self.available_actions = np.arange(self.env.action_space.n)

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
        return self.g_value_dict[current_state]

    # def get_action(self, obs):
    #     '''
    #     Get the action based on the observation
    #     :param obs:
    #     :return:
    #     '''
    #     utility_list = np.zeros(self.env.action_space.n)  # initialize the utility list to 0
    #
    #
    #     for action in range(self.env.action_space.n):
    #         print("trying action: ", action)
    #         # Obtain the transitionable for each action. (Avoid the wall)
    #         # Zero if the action is not transitionable
    #         state_transitionable = self.state_transitionable_dict[(tuple(obs), action)]
    #         if not state_transitionable:
    #             print("action: ", action, " is not transitionable at state: ", obs)
    #             utility_list[action] = 0
    #             continue
    #
    #         # get new state (after taking the action)
    #         new_state = deepcopy(obs)
    #         if action == 0:
    #             new_state[1] -= 1
    #         elif action == 1:
    #             new_state[1] += 1
    #         elif action == 2:
    #             new_state[0] += 1
    #         elif action == 3:
    #             new_state[0] -= 1
    #         else:
    #             raise Exception('Invalid action')
    #         # check if the new state is out of bound
    #         if new_state[0] < 0 or new_state[0] > self.env.maze_size[0] or \
    #                 new_state[1] < 0 or new_state[1] > self.env.maze_size[1]:
    #             print("action: ", action, " is out of bound at state: ", obs)
    #             utility_list[action] = 0
    #             continue
    #
    #         # TODO: Consider Dijkstra's algorithm to find the shortest path. (done)
    #         # g_value is the shortest cost/distance from the start state to the current state
    #         g_value = self.calculate_g_value(tuple(obs), tuple(new_state)) + 0.001 # add a small value to avoid divide by 0
    #         # h_value is the heuristic value (the Manhattan distance from the current state to the goal state)
    #         h_value = self.calculate_manhattan_distance(new_state, self.goal_state) + 0.001 # add a small value to avoid divide by 0
    #         # Calculate the A* value (A* algorithm) for each new state
    #         f_value = 0.5 * g_value + 1.5 * h_value
    #         print("f_value: ", f_value, " h_value: ", h_value, " g_value: ", g_value)
    #
    #         # Calculate the utility for each action
    #         utility = state_transitionable * (1 / f_value)
    #         utility_list[action] = utility
    #     print("obs: ", obs)
    #     print("utility_list: ", utility_list, "probability: ", utility_list / sum(utility_list))
    #     best_action = np.random.choice(self.env.action_space.n, p=utility_list / sum(utility_list))
    #     # best_action = np.argmax(utility_list)
    #     print("best_action: ", best_action)
    #     return best_action


    # def softmax(self, utilities, temperature=1.0):
    #     print("utilities: ", utilities, type(utilities))
    #     exp_utilities = np.exp(utilities / temperature)
    #     probabilities = exp_utilities / np.sum(exp_utilities)
    #     print("probabilities: ", probabilities, type(probabilities))
    #     return probabilities

    def softmax(self, utilities, temperature=0.5):
        utilities = utilities - np.max(utilities)  # Subtract max for numerical stability and avoid 'nan' after exponent
        exp_utilities = np.exp(utilities / temperature)
        probabilities = exp_utilities / np.sum(exp_utilities)
        return probabilities

    def get_action(self, obs):
        '''
        Get the action based on the observation.
        :param obs:
        :return:
        '''

        utility_list = self.get_utility_list(obs)


            # get new state (after taking the action)
            # new_state = deepcopy(obs)
            # if action == 0:
            #     new_state[1] -= 1
            # elif action == 1:
            #     new_state[1] += 1
            # elif action == 2:
            #     new_state[0] += 1
            # elif action == 3:
            #     new_state[0] -= 1
            # else:
            #     raise Exception('Invalid action')
            #
            # # check if the new state is out of bound
            # if new_state[0] < 0 or new_state[0] > self.env.maze_size[0] or \
            #         new_state[1] < 0 or new_state[1] > self.env.maze_size[1]:
            #     print("action: ", action, " is out of bound at state: ", obs)
            #     utility_list[action] = 0
            #     continue

            # explore to find the shortest path
            # g_value = self.calculate_g_value(tuple(obs), tuple(new_state)) + 0.001  # add a small value to avoid divide by 0
            # h_value = self.calculate_manhattan_distance(new_state, self.goal_state) + 0.001
            # f_value = 1.5*g_value + 0.5*h_value + self.state_visited[tuple(new_state)] * 10

            # Calculate the utility for each action
            # utility = state_transitionable * (1 / f_value)
            # utility_list[action] = utility

            # Test new utility function
            # utility_list[action] = self.calculate_manhattan_distance(self.start_state, new_state) + 0.001
            # calculate distance from new_state to the goal state


        # best_action = np.random.choice(self.env.action_space.n, p=utility_list / sum(utility_list))   # np fixed seed
        # utility_list *= 0.25
        action_prob = self.softmax(utility_list)
        action_prob = np.array(action_prob)
        best_action = random.choices(self.available_actions, weights=action_prob, k=1)[0]  #  unfixed seed
        # best_action = random.choices(self.available_actions, weights=utility_list / sum(utility_list), k=1)[0]  # unfixed seed
        return best_action

    def get_utility_list(self, obs):
        self.state_visited[tuple(obs)] = True
        utility_list = np.zeros(self.env.action_space.n)
        for action in range(self.env.action_space.n):
            state_transitionable = self.state_transitionable_dict[(tuple(obs), action)]
            # If there is a wall, action is not transitionable, set the utility of that action to 0
            if not state_transitionable:
                utility_list[action] = 0
                continue

            new_state = self.get_new_state(tuple(obs), action)
            if new_state == tuple(obs):
                utility_list[action] = 0
                continue
            utility_list[action] = 1 / (self.calculate_manhattan_distance(new_state, self.goal_state) + 0.001)
        return utility_list

    def get_DT_action_prob(self, obs):
        utility_list = self.get_utility_list(obs)
        action_prob = utility_list / sum(utility_list)
        return action_prob


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






    #     self.state_visited[tuple(obs)] = True

    # def get_action(self, obs):
    #     '''
    #     Get the action based on the observation. Use DFS to find the shortest path. If more than one step is needed, add the action to the action list.
    #     :param obs:
    #     :return:
    #     '''
    #
    #     obs = tuple(obs)  # convert obs to tuple
    #     print("obs: ", obs, "action_list: ", self.action_list)
    #
    #     # If there are actions left in the action_list, execute them first
    #     if self.action_list:
    #         return self.action_list.pop(0)
    #
    #     # If action_list is empty, find a new path using DFS
    #     stack = [(obs, action) for action in range(self.env.action_space.n)]    # initialize the stack with all possible actions from the current state
    #     print("obs: ", obs, "stack: ", stack, "visited: ", self.visited, "goal_state: ", self.goal_state, "state_transitionable_dict: ", self.state_transitionable_dict)
    #
    #     while stack:
    #         state, action = stack.pop()
    #         if state in self.visited:
    #             continue
    #
    #         self.visited.add(state)
    #
    #         # If the goal state is reached, convert the stack to action_list
    #         if state == self.goal_state:
    #             self.action_list = [action for _, action in stack]
    #             return self.action_list.pop(0)
    #
    #         # Add valid and unvisited states to the stack
    #         for action in range(self.env.action_space.n):
    #             new_state = self.get_new_state(state, action)
    #             if self.is_valid_state(new_state) and new_state not in self.visited:
    #                 stack.append((new_state, action))
    #
    #     # If there is no path to the goal, check unvisited neighbors. If no unvisited neighbors, return random action
    #     for action in range(self.env.action_space.n):
    #         new_state = self.get_new_state(obs, action)
    #         if self.is_valid_state(new_state) and new_state not in self.visited:
    #             self.action_list = [action]
    #             return self.action_list.pop(0)
    #
    #     return np.random.randint(self.env.action_space.n)

    def get_action_v2(self, obs):
        obs = tuple(obs)  # convert obs to tuple

        # add the current state to the visited set and remove it from the unvisited_neighbors set if it is in the set
        self.visited.add(obs)
        if obs in self.unvisited_states:
            self.unvisited_states.remove(obs)

        # check neighbors and add unvisited neighbor to the self.unvisited_states for later exploration
        for action in range(self.env.action_space.n):
            new_state = self.get_new_state(obs, action)
            if new_state not in self.visited:
                self.unvisited_states.add(new_state)
                self.unvisited_states_pre.add(obs)  # add the state where its neighbor has unvisited states

        print("obs: ", obs, "action_list: ", self.action_list)

        # If there are actions left in the action_list, execute them first
        if self.action_list:
            return self.action_list.pop(0)

        # if the goal state can be reached from the current state, find the shortest path to the goal state
        action_list_to_goal = self.get_action_list_to_goal(obs, self.goal_state)
        if action_list_to_goal is not None:
            self.action_list = action_list_to_goal
            return self.action_list.pop(0)

        print("obs: ", obs, "visited: ", self.visited, "goal_state: ", self.goal_state, 'unvisited_states: ', self.unvisited_states,
              "state_transitionable_dict: ", self.state_transitionable_dict)

        # if there are unvisited neighbors, explore one of them
        for action in range(self.env.action_space.n):
            new_state = self.get_new_state(obs, action)
            if new_state not in self.visited:
                return action

        # # If there is no unvisited neighbors, pick a state from the unvisited_states set and explore it
        # if self.unvisited_states:
        #     state_to_explore = self.unvisited_states.pop()
        #     self.action_list = self.get_action_list_to_goal(obs, state_to_explore)
        #     if self.action_list is not None:
        #         return self.action_list.pop(0)

        # If there is no unvisited neighbors, pick a state from the unvisited_states_pre set and explore its neighbors
        if self.unvisited_states_pre:
            state_to_explore = self.unvisited_states_pre.pop()
            for action in range(self.env.action_space.n):
                new_state = self.get_new_state(state_to_explore, action)
                if new_state not in self.visited:
                    return action

        # If there is no unvisited states, return random action
        print("random action")
        return np.random.randint(self.env.action_space.n)


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

        # update the state transitionable dictionary
        if tuple(new_obs) == self.old_state:
            self.state_transitionable_dict[(tuple(new_obs), action)] = False
            self.g_value_dict[(tuple(new_obs))] = float('inf')  # set the g_value to infinity
        else:
            if tuple(new_obs) not in self.visited:
                self.previous_state_dict[tuple(new_obs)] = self.old_state  # set the previous state
            self.old_state = tuple(new_obs)

        if tuple(new_obs) == self.goal_state:
            self.state_visited = defaultdict(lambda: False)  # reset the state_visited dictionary


    def get_action_v3(self, obs):
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


    def update_observation_v3(self, obs, action, new_obs, reward):
        '''
        Update the observation
        :param obs: old state
        :param action: action
        :param new_obs: new state
        :param reward: reward
        :return:
        '''
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




        # if tuple(new_obs) == self.goal_state:
        #     (node, path) = self.queue.get(tuple(new_obs))

        pass

