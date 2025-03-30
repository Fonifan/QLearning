import json
import random
import gymnasium as gym
import minigrid.core
import minigrid.core.constants
from gymnasium import Env, Space
from minigrid.minigrid_env import MiniGridEnv
import matplotlib.pyplot as plt

MAX_STEPS = 1000
MAX_EPISODES = 1000
NUM_ACTIONS = 3

def get_state(observation):
    image = observation['image']
    direction = observation.get('direction', None)
    return str(image.tolist()) + "|" + str(direction)

def add_state_to_Q(Q, state, A):
    if state not in Q:
        Q[state] = [0] * NUM_ACTIONS

def maxQ(state, Q):
    if state not in Q:
        return 0
    return max(range(len(Q[state])), key=lambda a: Q[state][a])

def select_action(Q, state, A, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        return maxQ(state, Q)
    
def initialize_Q():
    Q = {}
    return Q

def calculate_reward(steps, max_steps):
    return 1 - 0.9 * (steps / max_steps)

def q_learning(env: Env, alpha: float, epsilon: float, gamma: float, N_episodes: int, Q=None, baseline_reward=0):
    action_space = env.action_space
    epsilon_decay = 0.99
    epsilon_min = 0.01
    if Q is None:
        Q = initialize_Q()

    rewards = []
    steps_history = []
    for i in range(N_episodes):
        observation, _ = env.reset(seed=42)
        state = get_state(observation)
        add_state_to_Q(Q, state, action_space)
        terminated = False
        steps = 0
        while not terminated:
            a = select_action(Q, state, action_space, epsilon)
            next_ob, reward, terminated, truncated, info = env.step(a)
            steps += 1
            if steps > MAX_STEPS:
                reward = 0
                rewards.append(reward)
                steps_history.append(steps)
                print(f"Episode {i} truncated.")
            if terminated:
                reward = calculate_reward(steps, MAX_STEPS) + baseline_reward
                rewards.append(reward)
                steps_history.append(steps)
                print(f"Episode {i} terminated with {steps} steps, {reward} reward.")
            next_state = get_state(next_ob)
            add_state_to_Q(Q, next_state, action_space)
            
            best_next = max(Q[next_state])
            Q[state][a] = Q[state][a] + alpha * (reward + gamma * best_next - Q[state][a])
            
            state = next_state
            if steps > MAX_STEPS:
                break
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print("Epsilon: ", epsilon)
    return Q, rewards, steps_history

if __name__ == "__main__":
    env_name = "MiniGrid-Empty-16x16-v0"
    env = gym.make(env_name, max_episode_steps=MAX_STEPS)
    learned_Q, rewards, steps = q_learning(env, 0.1, 1, 0.99, MAX_EPISODES)
    print("Learned Q-table:")
    with open("learned_Q.json", "w") as f:
        json.dump(learned_Q, f, indent=4)
    env.close()

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.show()

    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps over Episodes')
    plt.show()
    
    env = gym.make(env_name, render_mode="human", max_episode_steps=MAX_STEPS)
    learned_Q, _ = q_learning(env, 0.1, 0, 0.99, MAX_EPISODES, learned_Q)
    env.close()
