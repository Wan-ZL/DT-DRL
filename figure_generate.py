'''
Project     : DT-DRL 
File        : figure_generate.py
Author      : Zelin Wan
Date        : 1/10/24
Description : 
'''
import glob
import os
from multiprocessing import Process
import tensorflow as tf
import numpy as np
from scipy.interpolate import make_interp_spline
from matplotlib import pyplot as plt
import matplotlib as mpl

def get_value_from_file(file_path, tag_name):
    value_set = []
    for summary_set in tf.compat.v1.train.summary_iterator(file_path):
        for value in summary_set.summary.value:
            if value.tag == tag_name:
                value_set.append(value.simple_value)

    return value_set

def data_path_finder(agent_name, fix_seed=None, env_name='CustomMaze', discrete_version=3, transfer_time_point=0):
    if fix_seed is None:
        path_seed_name = 'RandomSeed'
    else:
        path_seed_name = 'FixSeed'

    data_path = './data/' + path_seed_name + '/' + 'tb_reduce/' + env_name + '/agent_' + agent_name + '/env_discrete_ver_' + str(discrete_version)
    if transfer_time_point != 0:
        data_path = data_path + '/transfer_time_point_' + str(transfer_time_point) + '/mean/'
    else:
        data_path = data_path + '/mean/'
    file_path = glob.glob(data_path + '/*')
    if file_path == []:
        raise ValueError("No file found in the path: ", data_path)
    return file_path



# Given a list of scheme and the tag of tensorboard, draw the figure of the tag for all the schemes. Each scheme is a curve.
def draw_all_scheme_tag(agent_name_set, tag, discrete_version, env_name):
    # Get path of all schemes
    scheme_path_set = {}
    epi_size = 1
    plt.figure(figsize=(figure_width, figure_high))
    lines = []
    for agent_id, agent_name in enumerate(agent_name_set):
        if agent_name == 'DT_PPO_2':
            transfer_time_point = 100
        else:
            transfer_time_point = 0
        scheme_path_set[agent_name] = data_path_finder(agent_name, fix_seed=123, env_name=env_name, discrete_version=discrete_version, transfer_time_point=transfer_time_point)

        tag_name = tag + "/agent: " + agent_name
        y_axis = get_value_from_file(scheme_path_set[agent_name][0], tag_name)
        x_axis = list(range(len(y_axis)))
        # smooth curve
        X_Y_Spline = make_interp_spline(range(len(y_axis)), y_axis)
        X_ = np.linspace(0, len(y_axis), 50)
        Y_ = X_Y_Spline(X_)
        # draw the figure
        plt.plot(x_axis, y_axis, linewidth=figure_linewidth, alpha=0.4, color=agent_color[agent_name])
        line = plt.plot(X_[:-2], Y_[:-2], label=agent2str[agent_name], linewidth=figure_linewidth, color=agent_color[agent_name])[0]
        # line = plt.plot(X_, Y_, label=agent2str[agent_name], linewidth=figure_linewidth, color=agent_color[agent_name])[0]
        print("line: ", line)
        lines.append(line)

    plt.xlabel("Episodes", fontsize=font_size)
    plt.ylabel(tag2str[tag], fontsize=font_size)
    if show_range_y is not None:
        plt.ylim(show_range_y)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    # plt.legend(fontsize=legend_size)
    plt.tight_layout()
    if os.path.exists("figures") is False:
        os.makedirs("figures")
    # convert agent_name_set to string for file name
    agent_name_set_str = str(agent_name_set[0])
    for agent_name in agent_name_set[1:]:
        agent_name_set_str += "_" + str(agent_name)
    plt.savefig("figures/" + env_name + "-Ver_" + str(discrete_version) + "-" + agent_name_set_str + ".pdf", format='pdf', dpi=figure_dpi)
    plt.savefig("figures/" + env_name + "-Ver_" + str(discrete_version) + "-" + agent_name_set_str + ".png", format='png', dpi=figure_dpi)
    plt.show()

    # draw legend
    # draw legend
    figlegend = plt.figure(figsize=(10.3, 0.6)) #plt.figure(figsize=(21.1, 0.7))
    # Ensure that lines is a flat list of Line2D objects
    figlegend.legend(handles=lines, loc='center', ncol=len(lines), fontsize=legend_size)
    figlegend.savefig("figures/" + env_name + "-Ver_" + str(discrete_version) + "-" + agent_name_set_str + "-legend.pdf",
                      format='pdf', dpi=figure_dpi)
    figlegend.savefig("figures/" + env_name + "-Ver_" + str(discrete_version) + "-" + agent_name_set_str + "-legend.png",
                      format='png', dpi=figure_dpi)
    figlegend.show()  # Might need to be removed or commented out, depending on the environment

    print("scheme_path_set: ", scheme_path_set)


def draw_sens_analysis(agent_name_set, tag, discrete_version_set, env_name):
    # draw the bar figure to show the sensitivity analysis
    # Get path of all schemes
    scheme_path_set = {}
    bar_width = 1/(len(agent_name_set) + 1)
    bars = []
    plt.figure(figsize=(figure_width, figure_high+1))
    for agent_id, agent_name in enumerate(agent_name_set):
        y_axis_avg = []
        for discrete_version in discrete_version_set:
            scheme_path_set[discrete_version] = data_path_finder(agent_name, fix_seed=123, env_name=env_name, discrete_version=discrete_version)
            tag_name = tag + "/agent: " + agent_name
            y_axis = get_value_from_file(scheme_path_set[discrete_version][0], tag_name)
            y_axis_avg.append(np.mean(y_axis))   # get the average of last 100 episodes
        print("y_axis_avg: ", y_axis_avg)
        bottom_value = -1.05
        y_axis_avg = np.array(y_axis_avg) - bottom_value
        discrete_version_set = np.array(discrete_version_set)
        bar = plt.bar(discrete_version_set + agent_id * bar_width - bar_width * (len(agent_name_set) - 1) / 2, y_axis_avg, width=bar_width, label=agent2str[agent_name], bottom=bottom_value, color=agent_color[agent_name], hatch=bar_pattern[agent_id])
        bars.append(bar)

    plt.xlabel("Maze Size ($m$)", fontsize=font_size)
    plt.ylabel(tag2str[tag], fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    # plt.legend(fontsize=legend_size)
    plt.tight_layout()
    if os.path.exists("figures") is False:
        os.makedirs("figures")
    plt.savefig("figures/" + env_name + "-" + agent_name + "-sens_analysis.pdf", format='pdf', dpi=figure_dpi)
    plt.savefig("figures/" + env_name + "-" + agent_name + "-sens_analysis.png", format='png', dpi=figure_dpi)
    plt.show()

    # draw legend
    figlegend = plt.figure(figsize=(10.3, 0.6)) #plt.figure(figsize=(21.1, 0.7))
    # Ensure that lines is a flat list of Line2D objects
    figlegend.legend(handles=bars, loc='center', ncol=len(bars), fontsize=legend_size)
    figlegend.savefig("figures/" + env_name + "-" + agent_name + "-sens_analysis-legend.pdf",
                      format='pdf', dpi=figure_dpi)
    figlegend.savefig("figures/" + env_name + "-" + agent_name + "-sens_analysis-legend.png",
                        format='png', dpi=figure_dpi)
    # figlegend.show()


def draw_running_time(agent_name_set, discrete_version_set, env_name):
    # Get path of all schemes
    scheme_path_set = {}
    bar_width = 1 / (len(agent_name_set) + 1)
    bars = []
    plt.figure(figsize=(figure_width, figure_high+1))
    for agent_id, agent_name in enumerate(agent_name_set):
        y_axis_avg = []
        for discrete_version in discrete_version_set:
            fix_seed = 123
            # get the running time
            if fix_seed is None:
                seed_name = 'RandomSeed'
            else:
                seed_name = 'FixSeed'
            data_path = './data/' + seed_name + '/' + env_name + '/agent_' + agent_name + '/env_discrete_ver_' + str(discrete_version)
            file_name = 'running_time.txt'
            file_path = data_path + '/' + file_name
            with open(file_path, 'r') as f:
                running_time = float(f.read())
            # get step number
            tag = "step_count"
            scheme_path_set[discrete_version] = data_path_finder(agent_name, fix_seed=fix_seed, env_name=env_name, discrete_version=discrete_version)
            tag_name = tag + "/agent: " + agent_name
            step_count_list = get_value_from_file(scheme_path_set[discrete_version][0], tag_name)
            step_count_total = np.sum(step_count_list)
            # get running time per step
            running_time_per_step = running_time / step_count_total
            y_axis_avg.append(running_time_per_step)
        discrete_version_set = np.array(discrete_version_set)
        bar = plt.bar(discrete_version_set + agent_id * bar_width - bar_width * (len(agent_name_set) - 1) / 2, y_axis_avg, width=bar_width, label=agent2str[agent_name], color=agent_color[agent_name], hatch=bar_pattern[agent_id])
        bars.append(bar)

    # draw the figure
    plt.xlabel("Maze Size ($m$)", fontsize=font_size)
    plt.ylabel("Running Time Per Step", fontsize=font_size)
    plt.xticks(fontsize=axis_size)
    plt.yticks(fontsize=axis_size)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))   # y axis use sceientific notation
    # plt.legend(fontsize=legend_size)
    plt.tight_layout()
    if os.path.exists("figures") is False:
        os.makedirs("figures")
    plt.savefig("figures/" + env_name + "-" + agent_name + "-running_time.pdf", format='pdf', dpi=figure_dpi)
    plt.savefig("figures/" + env_name + "-" + agent_name + "-running_time.png", format='png', dpi=figure_dpi)
    plt.show()

    # draw legend
    figlegend = plt.figure(figsize=(10.3, 0.6)) #plt.figure(figsize=(21.1, 0.7))
    figlegend.legend(handles=bars, loc='center', ncol=len(bars), fontsize=legend_size)
    figlegend.savefig("figures/" + env_name + "-" + agent_name + "-running_time-legend.pdf",
                      format='pdf', dpi=figure_dpi)
    figlegend.savefig("figures/" + env_name + "-" + agent_name + "-running_time-legend.png",
                        format='png', dpi=figure_dpi)
    figlegend.show()



# setting for figure
font_size = 25  # 25
figure_high = 5 #5 # 6
figure_width = 7.5
figure_linewidth = 3
figure_dpi = 100
legend_size = 18  # 18
axis_size = 15
marker_size = 12
marker_list = ["p", "d", "v", "x", "s", "*", "1", "."]
bar_pattern = ["|", "\\", "/", ".", "-", "+", "*", "x", "o", "O"]
linestyle_list = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]
default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
# assign the color for each agent
agent_color = {"DT": default_colors[0], "PPO_2": default_colors[1], "FPER_PPO": default_colors[2], "DT_PPO_3": default_colors[3], "IL_PPO": default_colors[4], "TL_PPO": default_colors[5]}
strategy_number = 8
max_x_length = 60
use_legend = False
show_range_y = None # [-1.5, 1.5]
tag2str = {"accumulated_reward": "Accumulated Reward"}
agent2str = {"DT": "DT", "PPO_2": "PPO", "DT_PPO_3": "DT-guided PPO", "TL_PPO": "TL PPO", "FPER_PPO": "SE PPO", "IL_PPO": "IL PPO"}

if __name__ == '__main__':
    fix_seed = None
    discrete_version_set = [3, 4, 5, 6, 7, 8] # [3, 4, 5, 6, 7, 8]


    tag_set = ["accumulated_reward"]
    agent_name_set = ['DT', 'PPO_2', 'TL_PPO', 'FPER_PPO', 'DT_PPO_3'] # for SlightlyModifiedCartPole: ['DT', 'PPO_2', 'FPER_PPO', 'IL_PPO', 'DT_PPO_3']. For CustomMaze: ['DT', 'PPO_2', 'TL_PPO', 'FPER_PPO', 'DT_PPO_3']
    env_name = 'CustomMaze'  # choose environment from 'SlightlyModifiedCartPole', 'CustomMaze'
    # show the figure of training process
    # for discrete_version in discrete_version_set:
    #     for tag in tag_set:
    #         p1 = Process(target=draw_all_scheme_tag, args=(agent_name_set, tag, discrete_version, env_name))
    #         p1.start()

    # show the figure of sensitivity analysis
    draw_sens_analysis(agent_name_set, tag_set[0], [3, 4, 5, 6, 7, 8], 'CustomMaze')

    # show the figure of training time per step
    draw_running_time(agent_name_set, [3, 4, 5, 6, 7, 8], 'CustomMaze')