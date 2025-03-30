import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from game.src.card import Card
from game.src.game import Game, actions, ranks, suits

NO_CARD_ID = -1
MAX_HAND_SIZE = 11
MAX_TABLE_PAIRS = 6
MAX_STEPS = 1000
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x

def select_action(state, valid_actions, epsilon, policy_net):
    if random.random() < epsilon:
        return random.choice(valid_actions)
    encoded = encode_state(state)
    encoded = encoded.unsqueeze(0)
    q_values = policy_net(encoded)
    invalid_mask = torch.ones(q_values.size(), dtype=torch.bool)
    for action in valid_actions:
        invalid_mask[0][action] = False
    q_values[invalid_mask] = -1e9  # effectively -∞
    return torch.argmax(q_values).item()

def card_to_int(card):
    suit_index = suits.index(card.suit)
    return card.rank * len(suits) + suit_index

def encode_state(state):
    player_hand = state['cards']
    hand_encoded = np.full(shape=(MAX_HAND_SIZE,), fill_value=NO_CARD_ID, dtype=np.int32)
    for i, card in enumerate(player_hand[:MAX_HAND_SIZE]):
        hand_encoded[i] = card_to_int(card)

    table_cards = np.full(shape=(MAX_TABLE_PAIRS * 2,), fill_value=NO_CARD_ID, dtype=np.int32)
    for i, pair in enumerate(state['table'][:MAX_TABLE_PAIRS]):

        if len(pair) > 0 and pair[0] is not None:
            table_cards[2*i] = card_to_int(pair[0])    # attacking card
        if len(pair) > 1 and pair[1] is not None:
            table_cards[2*i + 1] = card_to_int(pair[1])  # defending card

    if isinstance(state['trump'], Card):
        trump_encoded = card_to_int(state['trump'])
    elif isinstance(state['trump'], str):
        trump_encoded = suits.index(state['trump'])  # e.g. 0..3
    else:
        trump_encoded = NO_CARD_ID

    discard_count = len(state['discarded'])

    state_array = np.concatenate([ # TODO add discarded cards to state
        hand_encoded,
        table_cards,
        np.array([trump_encoded, discard_count], dtype=np.int32)
    ])

    return torch.FloatTensor(state_array)

def card_to_onehot(card):
    card_onehot = torch.zeros(24)
    card_index = ranks.index(card.rank) * 4 + suits.index(card.suit)
    card_onehot[card_index] = 1.0
    return card_onehot

def states_to_tensor(states):
    tensors = []

    for state in states:
        tensors.append(encode_state(state))

    return torch.stack(tensors)

def calculate_reward(done, episode_steps, max_steps):
    if done:
        return 1.0 - 0.9 * (episode_steps / max_steps)
    else:
        return 0
    
def optimize_model(memory, policy, target, optimizer):
    if len(memory) >= batch_size:
        batch = random.sample(memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = states_to_tensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = states_to_tensor(next_state_batch)
        done_batch = torch.IntTensor(done_batch)

        q_values = policy(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
            max_next_q_values = target(state_batch).max(1)[0]
            target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    output_dim = len(actions)
    input_dim = MAX_HAND_SIZE + MAX_TABLE_PAIRS * 2 + 2
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)

    opponent_policy = DQN(input_dim, output_dim)
    opponent_target = DQN(input_dim, output_dim)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    opponent_target.load_state_dict(policy_net.state_dict())
    opponent_target.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    opponent_optimizer = optim.Adam(opponent_policy.parameters(), lr=learning_rate)

    memory = deque(maxlen=memory_size)
    opponent_memory = deque(maxlen=memory_size)
    env = Game()

    for episode in range(episodes):
        state = env.start()
        episode_reward = 0
        done = False
        episode_steps = 0
        while not done:
            valid_actions = env.get_valid_actions()
            action = select_action(state, valid_actions, epsilon, policy_net)
            next_state, reward , done, truncated, _ = env.step(action)

            if done:
                opponent_reward = -1
            if not done:
                valid_actions = env.get_valid_actions()
                opponent_action = select_action(next_state, valid_actions, epsilon, opponent_policy)
                opponent_next_state, opponent_reward , opponent_done, truncated, _ = env.step(opponent_action)

            if opponent_done:
                done = True
                reward = -1

            memory.append((state, action, reward, next_state, done))
            opponent_memory.append((next_state, opponent_action, opponent_reward, opponent_next_state, opponent_done))

            if episode_steps >= MAX_STEPS:
                done = True
            if done:
                print(f"Episode {episode} finished after {episode_steps} steps with reward {reward} opponent {opponent_reward}")
            
            state = next_state
            
            optimize_model(memory, policy_net, target_net, optimizer)
            optimize_model(opponent_memory, opponent_policy, opponent_target, opponent_optimizer)

            if reward > 0:
                opponent_policy.load_state_dict(policy_net.state_dict())
                opponent_target.load_state_dict(target_net.state_dict())
                opponent_memory = memory.copy()
                opponent_memory.clear()
            if opponent_reward > 0:
                policy_net.load_state_dict(opponent_policy.state_dict())
                target_net.load_state_dict(opponent_target.state_dict())
                memory = opponent_memory.copy()
                memory.clear()

            if steps_done % target_update_freq == 0:
                print(f"Updating target network at step {steps_done}")
                target_net.load_state_dict(policy_net.state_dict())
                opponent_target.load_state_dict(opponent_policy.state_dict())
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
