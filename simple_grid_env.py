import sys
from time import sleep
import gymnasium as gym
import numpy as np

class SimpleGrid(gym.Env):
    def __init__(self, grid_size=5, max_steps=1024, obstacle=None, render_mode=None):
        super(SimpleGrid, self).__init__()
        self.grid_size = grid_size
        self.target_pos = (grid_size - 1, grid_size - 1)
        self.agent_pos = (0, 0)
        self.obstacle = obstacle if obstacle is not None else []
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(grid_size, grid_size), dtype=np.int8)
        self.state = None
        self.done = False
        self.truncated = False
        self.action_idx = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),    # Right,
            4: (-1, -1), # Up-Left
            5: (-1, 1),  # Up-Right
            6: (1, -1),  # Down-Left
            7: (1, 1)    # Down-Right
        }
        self.pretty_action = {
            0: "↑",
            1: "↓",
            2: "←",
            3: "→",
            4: "↖",
            5: "↗",
            6: "↙",
            7: "↘"
        }
        self.action_space = gym.spaces.Discrete(len(self.action_idx))
        self.steps = 0
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.max_distance = self.calculate_distance((0, 0), self.target_pos)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.agent_pos = (0, 0)
        self.state[self.agent_pos] = 1
        self.state[self.target_pos] = 2
        for obs in self.obstacle:
            if obs != self.agent_pos and obs != self.target_pos:
                self.state[obs] = 3
        self.done = False
        self.truncated = False
        self.steps = 0
        self.reward = 0
        return self._get_obs(), {}
    
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        self.steps += 1
        dx, dy = self.action_idx[action]
        old_pos = self.agent_pos
        new_x = max(0, min(self.grid_size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.grid_size - 1, self.agent_pos[1] + dy))
        if self.state[new_x, new_y] == 3:  # Obstacle
            new_x, new_y = self.agent_pos
        self.state[self.agent_pos] = 0
        self.agent_pos = (new_x, new_y)
        self.state[self.agent_pos] = 1
        self.render(self.render_mode)
        self.done = self.is_done()
        self.truncated = self.is_truncated()
        return self._get_obs(), self.calculate_reward(old_pos), self.done, self.truncated, {}

    def is_done(self):
        if self.agent_pos == self.target_pos:
            return True
        return False
    
    def is_truncated(self):
        if self.steps >= self.max_steps:
            return True
        return False

    def render(self, mode='ansi'):
        if mode != 'ansi':
            return
        grid = np.copy(self.state)
        grid[self.agent_pos] = 1
        grid[self.target_pos] = 2
        print(grid)
        sleep(0.2)
    
    def _get_obs(self):
        return self.state.copy()

    def calculate_reward(self, old_pos):
        if self.agent_pos == self.target_pos:
            return 1 - 0.9 * (self.steps / self.max_steps)
        else:
            pos_diff = self.calculate_intermediate_reward(self.agent_pos) - self.calculate_intermediate_reward(old_pos)
            steps_penalty = (np.sign(pos_diff) if pos_diff != 0 else 1) * ((self.steps /  self.max_steps) / self.max_steps)
            return pos_diff - steps_penalty
        
    def calculate_intermediate_reward(self, pos):
        return (self.max_distance - self.calculate_distance(pos, self.target_pos) )/ self.max_distance

    def calculate_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def pretty_print_action(self, action):
        return self.pretty_action.get(action, "Unknown Action")