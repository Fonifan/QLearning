import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import minigrid
import matplotlib.pyplot as plt

MAX_STEPS = 50_000
NUM_ACTIONS = 6
NUM_DIRECTIONS = 4
ENVIRONMENT_ID = "MiniGrid-DoorKey-16x16-v0"
class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2)
        self.pool = nn.MaxPool2d(2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(16 * 4 * 4 + NUM_DIRECTIONS, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(64, output_dim)

    def forward(self, image, direction):
        x = torch.relu(self.conv1(image))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        cat_state = torch.cat((x, direction), dim=1)
        x = torch.relu(self.fc1(cat_state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x

def select_action(state, epsilon, policy_net):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    image, direction = state_to_tensor(state)
    image = image.unsqueeze(0)
    direction = direction.unsqueeze(0)
    q_values = policy_net(image, direction)
    return torch.argmax(q_values).item()

def state_to_tensor(state):
    image = state['image']
    image = torch.FloatTensor(image).permute(2, 0, 1) # Change to (C, H, W) for Conv2d
    direction = state['direction']
    direction_one_hot = torch.zeros(NUM_DIRECTIONS)
    direction_one_hot[direction] = 1.0
    return torch.FloatTensor(image), direction_one_hot

def states_to_tensor(states):
    images = []
    directions = []
    for state in states:
        image, direction = state_to_tensor(state)
        images.append(image)
        directions.append(direction)
    return torch.stack(images), torch.stack(directions)

def calculate_reward(done, episode_steps, max_steps):
    if done:
        return 1.0 - 0.9 * (episode_steps / max_steps)
    else:
        return 0

if __name__ == "__main__":
    rewards_per_episode = []
    steps_done = 0
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.99
    batch_size = 64
    target_update_freq = 5000
    memory_size = 10000
    episodes = 1000

    output_dim = NUM_ACTIONS
    policy_net = DQN(output_dim)
    target_net = DQN(output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size)
    env = gym.make(ENVIRONMENT_ID, max_episode_steps=MAX_STEPS)

    for episode in range(episodes):
        state, info = env.reset(seed=42)
        episode_reward = 0
        done = False
        episode_steps = 0
        while not done:
            action = select_action(state, epsilon, policy_net)
            next_state, _ , done, truncated, _ = env.step(action)
            
            reward = calculate_reward(done, episode_steps, MAX_STEPS)
            if episode_steps >= MAX_STEPS:
                done = True
            if done:
                print(f"Episode {episode} finished after {episode_steps} steps with reward {reward}")
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            episode_reward += reward
            
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

                state_batch = states_to_tensor(state_batch)
                action_batch = torch.LongTensor(action_batch).unsqueeze(1)
                reward_batch = torch.FloatTensor(reward_batch)
                next_state_batch = states_to_tensor(next_state_batch)
                done_batch = torch.IntTensor(done_batch)

                q_values = policy_net(state_batch[0], state_batch[1]).gather(1, action_batch).squeeze()

                with torch.no_grad():
                    max_next_q_values = target_net(next_state_batch[0], next_state_batch[1]).max(1)[0]
                    target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

                loss = nn.MSELoss()(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % target_update_freq == 0:
                print(f"Updating target network at step {steps_done}")
                target_net.load_state_dict(policy_net.state_dict())
                torch.save(policy_net.state_dict(), "latest_policy_net.pt")


            steps_done += 1
            episode_steps += 1

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        
        rewards_per_episode.append(episode_reward)

    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Rewards')
    plt.show()

    torch.save(policy_net.state_dict(), "dqn.pt")
