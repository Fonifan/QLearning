import copy
import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np

from make_heatmap_video import create_vid
from simple_grid_env import SimpleGrid

MAX_STEPS = 2700
MAX_EPISODES = 700
historic_Qs = []
def get_state(observation):
    return observation

def select_action(Q, state, action_space, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_space.n - 1)
    else:
        return get_maxA(state, Q)
    
def initialize_Q(env):
    Q = {}
    for i in range(env.observation_space.shape[0]):
        for j in range(env.observation_space.shape[1]):
            Q[(i, j)] = [0] * env.action_space.n
    Q[env.target_pos] = [1] * env.action_space.n
    return Q

def find_player(state):
    player_pos = None
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i, j] == 1:
                player_pos = (i, j)
                return player_pos
    return player_pos

def get_maxQ(state, Q):
    player_pos = find_player(state)
    if player_pos is None:
        raise ValueError("Player position not found in state.")
    return np.max(Q[player_pos])

def get_maxA(state, Q):
    player_pos = find_player(state)
    if player_pos is None:
        raise ValueError("Player position not found in state.")
    return np.argmax(Q[player_pos])

def q_learning(env: SimpleGrid, alpha: float, epsilon: float, gamma: float, N_episodes: int, Q=None, baseline_reward=0):
    epsilon_decay = 0.99
    epsilon_min = 0.01
    if Q is None:
        Q = initialize_Q(env)

    rewards = []
    steps_history = []
    for i in range(N_episodes):
        observation, _ = env.reset(seed=42)
        state = get_state(observation)
        done = False
        while not done:
            # env.render()
            a = select_action(Q, state, env.action_space, epsilon)
            next_ob, reward, terminated, truncated, info = env.step(a)
            if truncated:
                rewards.append(reward)
                steps_history.append(env.steps)
                print(f"Episode {i} truncated.")
                done = True
            if terminated:
                rewards.append(reward)
                steps_history.append(env.steps)
                print(f"Episode {i} terminated with {env.steps} steps, {reward} reward.")
                done = True
            next_state = get_state(next_ob)
            
            best_next = get_maxQ(next_state, Q)
            Q[find_player(state)][a] = Q[find_player(state)][a] + alpha * (reward + gamma * best_next - Q[find_player(state)][a])
            state = next_state
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print("Epsilon: ", epsilon)
        write_Q(Q)
    return Q, rewards, steps_history

def write_Q(Q):
    historic_Qs.append(copy.deepcopy(Q))

if __name__ == "__main__":
    # env = SimpleGrid(grid_size=20, max_steps=MAX_STEPS, obstacle=[(10, 10), (11, 10), (12, 10), (13, 10), (14,10), (10, 9), (10, 8), (10, 7), (10, 11), (10, 12), (10, 13), (9, 10), (8, 10), (7, 10), (6, 10), (5, 10), (4, 10), (3, 10), (2, 10), (1, 10)])
    env = SimpleGrid(grid_size=20, max_steps=MAX_STEPS)
    learned_Q, rewards, steps = q_learning(env, 0.1, 1, 0.99, MAX_EPISODES)

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.show()

    print("Generating video...")
    create_vid(historic_Qs, env)
