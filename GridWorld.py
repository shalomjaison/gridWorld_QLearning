import gym
from gym import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self, grid):
        super(GridWorldEnv, self).__init__()
        self.grid = np.array(grid) # Much more efficient compared to Python-Lists
        self.actions = ['up', 'down', 'left', 'right']
        self.action_space = spaces.Discrete(len(self.actions))
        self.rows, self.cols = self.grid.shape
        # Generalizing the code 
        self.start = self.find_state('S')
        print(self.start)
        self.goals = self.find_states('G')
        self.death_states = self.find_states('X')
        self.action_probs_path1 = {
            'up': {'up': 1.0},
            'down': {'down': 1.0},
            'left': {'left': 1.0},
            'right': {'right': 1.0}
        }
        self.action_probs_path2 = {
            'up': {'up': 0.8, 'left': 0.10, 'right': 0.10},
            'down': {'down': 0.8, 'left': 0.10, 'right': 0.10},
            'left': {'left': 0.8, 'up': 0.10, 'down': 0.10},
            'right': {'right': 0.8, 'up': 0.10, 'down': 0.10}
        }
        self.observation_space = spaces.Discrete(self.rows * self.cols) # Unique States in the grid
        # spaces.Discrete(n) defines a space where each state can take on integer values from 0 to n-1
        # Each position in the grid is a unique state, and so the observation space has each of these states
        self._max_episode_steps = 3000
        self.timesteps = 0
        self.decisions = 20
        # self.path = []
        self.reset()
    
    def find_state(self, state_char):
        result = np.where(self.grid == state_char)
        return (result[0][0], result[1][0]) if result[0].size > 0 else None
    
    def find_states(self, state_char):
        result = np.where(self.grid == state_char)
        return list(zip(result[0], result[1])) if result[0].size > 0 else []
    
    def _get_obs(self):
        # Map each unique position in a grid environment to a specific state
        # Unique Identifier
        return self.current_position[0] * self.cols + self.current_position[1]
    
    def reset(self):
        self.current_position = self.start
        self.timesteps = 0
        self.decisions = 20
        # self.path = [self.current_position] 
        return self._get_obs()

    def is_path_2(self, position):
        x, y = position
        return 1 <= x < self.rows and 2 < y < self.cols  # For Grid 2
    
    def step(self, action, decision=True):
        self.timesteps += 1
        self.decisions -= decision
        if (action not in range(len(self.actions))):
            raise ValueError("Invalid Action: " + str(action))
        
        action_str = self.actions[action]
        action_probs = self.action_probs_path2 if self.is_path_2(self.current_position) else self.action_probs_path1
        actual_action = np.random.choice(self.actions, p=[action_probs[action_str].get(a, 0) for a in self.actions])

        x, y = self.current_position
        if actual_action == 'up':
            x -= 1
        elif actual_action == 'down':
            x += 1
        elif actual_action == 'left':
            y -= 1
        elif actual_action == 'right':
            y += 1

        # Problem because of Wall it may cause a few values to repeat
        if 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] != 'W':
            self.current_position = (x, y)
            # self.path.append(self.current_position)
        reward = -1
        done = False

        if self.current_position in self.goals:
            print("LETS GOOOOOOO121212", self.current_position)
            reward = 10  # Reward for reaching the goal
            done = True
        elif self.current_position in self.death_states:
            reward = -10  # Penalty for hitting a death state
            done = True
        
        
        return self._get_obs(), reward, done

