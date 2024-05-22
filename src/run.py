import sys
#sys.path.insert(0, "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project")
import numpy as np
#import class methos .py files 
import unittest
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import csv

device = "/gpu:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
print(f"Running on {device}")
runMultipleParameters = False
training_started = False

def show_progress(label, full, prog):
    sys.stdout.write("\r{0}: {1}%  [{2}{3}]".format(label, prog, "█"*full, " "*(30-full)))
    sys.stdout.flush()
    
# def plot_heatMap(q):
#     state_labels = ["START_AREA", "GOAL_AREA", "WINNING_AREA", "DANGER_AREA", "SAFE_AREA", "DEFAULT_AREA"]
#     action_labels = ["STARTING_ACTION", "DEFAULT_ACTION", "INSIDE_GOAL_AREA_ACTION", "ENTER_GOAL_AREA_ACTION", "ENTER_WINNING_AREA_ACTION", "STAR_ACTION", "MOVE_INSIDE_SAFETY_ACTION", "MOVE_OUTSIDE_SAFETY_ACTION", "KILL_PLAYER_ACTION", "DIE_ACTION", "NO_ACTION"]

#     fig, ax = plt.subplots()
#     im = ax.imshow(q.Q_table, cmap='viridis')

#     ax.set_xticks(np.arange(len(action_labels)))
#     ax.set_yticks(np.arange(len(state_labels)))
#     ax.set_xticklabels(action_labels)
#     ax.set_yticklabels(state_labels)

#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     cbar = ax.figure.colorbar(im, ax=ax) # Add colorbar to show color scale
#     cbar.ax.set_ylabel("Q-values", rotation=-90, va="bottom") # Add label to colorbar

#     for i in range(len(state_labels)):
#         for j in range(len(action_labels)):
#             text = ax.text(j, i, int(q.Q_table[i, j] * 100), ha="center", va="center", color="w")

#     ax.set_title("Q-table")
#     fig.tight_layout()
#     plt.savefig('/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/images/heatmap.svg', format='svg', dpi=1200)
#     plt.show()


def startMethods():

    return 


def training(q, number_of_runs_for_training, q_player, after=0):
        
    if runMultipleParameters == False:
        progress = int(((i + 1) / number_of_runs_for_training) * 30)
        show_progress("training",progress, int(((i + 1) / number_of_runs_for_training) * 100))

    return 

def validation(q, number_of_runs_for_validation, q_player, after=0):
    wins = [0, 0, 0, 0]
    q.training = 0
    array_of_sum_of_rewards = []
    win_rate_list = []
    average_validate_win_rates = []  # List to hold average win rates
    for i in range(number_of_runs_for_validation):
        first_winner, sum_of_rewards, win_rate = playGame(q, q_player, training=False, current_game = i + after, after=after)
        array_of_sum_of_rewards.append(sum_of_rewards)
        win_rate_list.append(win_rate)
        q.reset()
        wins[first_winner] = wins[first_winner] + 1
        
        # ####################### Set for comparation: 50 games a 150 episodes #######################
        # if (i+1) % 50 == 0:
        #     average_win_rate = sum(win_rate_list[-75:]) / 75
        #     average_validate_win_rates.append(average_win_rate)
        
        ####################### Set for induvidual test: 20 games a 100 episodes #######################
        if (i+1) % 50 == 0:
            average_win_rate = sum(win_rate_list[-50:]) / 50
            average_validate_win_rates.append(average_win_rate)
            
            
        if runMultipleParameters == False:
            progress = int(((i + 1) / number_of_runs_for_validation) * 30)
            show_progress("validation", progress, int(((i + 1) / number_of_runs_for_validation) * 100))
        
    #print("\n")
    return wins, array_of_sum_of_rewards, win_rate_list, average_validate_win_rates



def run():
    # Parameters
    return

def save_data_and_parameters(win_rate_vec, explore_rate_vec, discount_factor_vec, learning_rate_vec, number_of_runs_for_training, number_of_runs_for_validation, actions_per_game, games_wins, average_win_rates):
    folder_path = os.path.join(os.getcwd(), "/Users/reventlov/Documents/Robcand/2. Semester/TAI/Exam/Ludo-Q-learning-project/src/data")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    data_file_path = os.path.join(folder_path, "data.npy")
    np.save(data_file_path, [win_rate_vec, actions_per_game, games_wins, average_win_rates])

    param_file_path = os.path.join(folder_path, "parameters.npy")
    np.save(param_file_path, [explore_rate_vec, discount_factor_vec, learning_rate_vec, number_of_runs_for_training, number_of_runs_for_validation])

class MyTestCase(unittest.TestCase):
    def test_something(self):
        with tf.device('/device:GPU:0'):
            self.assertEqual(True, run())


if __name__ == '__main__':
    unittest.main()

