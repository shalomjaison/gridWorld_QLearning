from GridWorld import GridWorldEnv
import numpy as np
import random

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    # Q-table is [num_states x num_actions]
    q_table = np.zeros((env.rows * env.cols, len(env.actions)))

    for curr_episode in range(num_episodes):
        curr_state = env.reset() # Back to initial state
        flag = False # A boolean flag to see if the episode is 'done'

        while not flag:
            # ϵ-greedy strategy
            # Exploit and Explore
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore
            else:
                action = np.argmax(q_table[curr_state, :]) # Exploit
            

            next_state, reward, flag = env.step(action) # Find the next state and the reward that the agent will get, if the agent takes the above action
            old_value = q_table[curr_state, action]
            next_max = np.max(q_table[next_state, :]) # Find the next best action-next_state pair
            
            # Principle of Q-Learning Function
            # alpha is the learning rate
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[curr_state, action] = new_value
            curr_state = next_state
    return q_table

def get_best_path(env, q_table):
    # Map out the best path from the q_table
    state = env.reset()
    path = [env.current_position]
    done = False

    while not done:
        action = np.argmax(q_table[state, :]) # Finds the next best action to take for the current state
        next_state, _, done = env.step(action)
        path.append(env.current_position)
        state = next_state

    return path


if __name__ == '__main__':

    # GRID 1
    # grid = [
#     ['-', '-', '-', '-', '-', '-', 'S', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', '-', 'X', 'W', 'W', 'W', 'W'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', '-', '-', 'X', 'W', 'W', 'W'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', '-', '-', 'X', 'W', 'W'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', 'G', '-', 'X', 'W'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', '-', '-', 'X'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', '-', '-'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', '-', '-', '-'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', '-', '-', '-'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', '-', '-', '-'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', '-', '-', '-'],
#     ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', '-', '-', '-'],
#     ['G', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', '-', '-', '-']
#    ]

    # GRID 2
    grid = [
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', 'S', '-', '-', '-'],
    ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'W', 'W', '-', 'W', 'W', 'W'],
    ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', '-', '-', 'X', 'W'],
    ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', '-', '-', 'X', 'W'],
    ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', '-', '-', 'X', 'W'],
    ['-', 'W', 'W', 'W', 'W', 'W', 'W', 'X', '-', '-', '-', 'X', 'W'],
    ['-', 'W', 'W', 'X', 'X', 'X', 'X', 'X', '-', '-', '-', 'X', 'W'],
    ['-', 'W', 'W', '-', '-', '-', '-', '-', '-', '-', '-', 'X', 'W'],
    ['-', '-', 'G', '-', '-', '-', '-', '-', '-', '-', '-', 'X', 'W'],
    ['w', 'W', 'W', '-', '-', '-', '-', '-', '-', '-', 'X', 'W', 'W'],
    ['w', 'W', 'W', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'W', 'W', 'W'],
    ]

    num_episodes = 2000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    env = GridWorldEnv(grid)

    # Train the agent using Q-Learning
    q_table = q_learning(env, num_episodes, alpha, gamma, epsilon)

    # Get the best path found by the agent
    best_path = get_best_path(env, q_table)
    print(q_table)
    print("Best path found by the agent:", best_path)