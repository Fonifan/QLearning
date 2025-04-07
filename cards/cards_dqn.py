import csv
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from card_env import CardDurakEnv, Action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 1000
MAX_HAND_SIZE = 20
MAX_TABLE_PAIRS = 6

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.hand_size = MAX_HAND_SIZE
        self.table_size = MAX_TABLE_PAIRS * 2 
        self.state_size = 3  # deck_size, trump, attacking
        
        self.card_embedding = nn.Embedding(37, 16, padding_idx=0)  # Assuming cards are 0-35
        self.state_embedding = nn.Linear(self.state_size, 16)
        
        self.hand_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
        self.table_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
        
        self.hand_fc = nn.Linear(32 * self.hand_size, 64)  # 32 = 16*2 (from concatenation)
        self.table_fc = nn.Linear(16 * self.table_size, 64)
        self.state_fc = nn.Linear(16, 32)

        # Output layers
        self.combine = nn.Linear(64 + 64 + 32, 128)
        self.hidden = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)
        
    def forward(self, x):
        hand = x[:, :self.hand_size].long()  # [batch, hand_size]
        
        pos = self.hand_size
        table = x[:, pos:pos+self.table_size].long()  # [batch, table_size]
        pos += self.table_size
        
        state = x[:, pos:]  # [batch, 3]
        
        hand_emb = self.card_embedding(hand)  # [batch, hand_size, 16]
        table_emb = self.card_embedding(table)  # [batch, table_size, 16]
        state_emb = self.state_embedding(state)  # [batch, 16]
        
        hand_att, _ = self.hand_attention(hand_emb, hand_emb, hand_emb)
        table_att, _ = self.table_attention(table_emb, table_emb, table_emb)
        
        hand_to_table, _ = self.cross_attention(hand_emb, table_emb, table_emb)
        
        hand_features = torch.cat([hand_att, hand_to_table], dim=2)
        hand_features = hand_features.flatten(start_dim=1)
        table_features = table_att.flatten(start_dim=1)
        
        hand_out = F.relu(self.hand_fc(hand_features))
        table_out = F.relu(self.table_fc(table_features))
        state_out = F.relu(self.state_fc(state_emb))
        
        combined = torch.cat([hand_out, table_out, state_out], dim=1)
        
        x = F.relu(self.combine(combined))
        x = F.relu(self.hidden(x))
        x = self.out(x)
        
        return x

def states_to_tensor(states):
    tensors = []
    for state in states:
        tensors.append(state_to_tensor(state).to(device))
    return torch.stack(tensors)

def state_to_tensor(state):

    deck_size = torch.FloatTensor([state["deck_size"]])
    trump = torch.FloatTensor([state["trump"]])
    attacking = torch.FloatTensor([state["attacking"]])
    
    # TODO include discard later
    return torch.cat([
        torch.IntTensor(state['hand']),
        torch.IntTensor(state['table'].flatten()),
        deck_size, trump, attacking
    ])

def select_action(state, valid_actions, epsilon, policy_net):
    if random.random() < epsilon:
        return random.choice(valid_actions)
    
    encoded = state_to_tensor(state).unsqueeze(0).to(device)
    q_values = policy_net(encoded)
    
    invalid_mask = torch.ones(q_values.size(), dtype=torch.bool)
    for action in valid_actions:
        invalid_mask[0][action] = False
    q_values[invalid_mask] = -1e9
    return torch.argmax(q_values).item()

def optimize_model(memory, policy, target, optimizer, batch_size, gamma):
    if len(memory) >= batch_size:
        batch = random.sample(memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = states_to_tensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        next_state_batch = states_to_tensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch).to(device)

        q_values = policy(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
            max_next_q_values = target(next_state_batch).max(1)[0]
            target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    return 0

def soft_update(source_net, target_net, tau=0.01):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def test(episode):
    output_dim = 22
    policy_net = DQN(input_dim, output_dim).to(device)
    try:
        policy_net.load_state_dict(torch.load("latest.card.dqn.pt", map_location=device))
        policy_net.eval()
    except Exception as e:
        print("Error loading model:", e)
        return

    env = CardDurakEnv()

    agent_wins = 0
    agent_losses = 0
    total_games = 200
    for i in range(total_games):
        state = env.reset()
        done = False
        while not done:
            valid_actions = env._get_valid_actions(player_id=1)
            try:
                player_input = random.choice(valid_actions)  # Random agent 
            except ValueError:
                continue
            if player_input not in valid_actions:
                continue

            state, reward, done, _, _ = env.step(player_input, player_id=1)
            if done:
                agent_losses += 1 if reward > 0 else 0
                agent_wins += 1 if reward < 0 else 0
                break


            valid_actions_opponent = env._get_valid_actions(player_id=2)
            opponent_action = select_action(state, valid_actions_opponent, epsilon=0, policy_net=policy_net)

            state, opp_reward, done, _, _ = env.step(opponent_action, player_id=2)
            if done:
                final_reward = -opp_reward
                agent_losses += 1 if final_reward > 0 else 0
                agent_wins += 1 if final_reward < 0 else 0
                break
    write_results_to_csv(episode, total_games, agent_wins, agent_losses)

def write_results_to_csv(episode, total_games, agent_wins, agent_losses):
    csv_file = "card_test_results.csv"
    win_rate = agent_wins / total_games if total_games > 0 else 0
    
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['Episode', 'Total Games', 'Agent Wins', 'Agent Losses', 'Win Rate'])
        
        writer.writerow([episode, total_games, agent_wins, agent_losses, f"{win_rate:.4f}"])


if __name__ == "__main__":
    rewards_per_episode = []
    steps_done = 0
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    target_update_freq = 1000
    memory_size = 10000
    episodes = 2000

    env = CardDurakEnv()

    output_dim = env.action_space.n
    input_dim = MAX_HAND_SIZE + MAX_TABLE_PAIRS * 2 + 3  # hand + table + deck_size + trump + attacking
    
    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)

    opponent_policy = DQN(input_dim, output_dim).to(device)
    opponent_target = DQN(input_dim, output_dim).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    opponent_target.load_state_dict(policy_net.state_dict())
    opponent_target.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    opponent_optimizer = optim.Adam(opponent_policy.parameters(), lr=learning_rate)

    memory = deque(maxlen=memory_size)
    opponent_memory = deque(maxlen=memory_size)

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_steps = 0
        
        while not done and episode_steps < MAX_STEPS:
            valid_actions = env._get_valid_actions(player_id=1)
            action = select_action(state, valid_actions, epsilon, policy_net)
            next_state, reward, done, truncated, _ = env.step(action, player_id=1)
            memory.append((state, action, reward, next_state, done))
            
            opponent_reward = 0
            opponent_done = False
            if not done:
                valid_actions = env._get_valid_actions(player_id=2)
                opponent_action = select_action(next_state, valid_actions, epsilon, opponent_policy)
                opponent_next_state, opponent_reward, opponent_done, truncated, _ = env.step(opponent_action, player_id=2)
                opponent_memory.append((next_state, opponent_action, opponent_reward, opponent_next_state, opponent_done))
                state = opponent_next_state
                if opponent_done:
                    done = True
                    reward = -opponent_reward  
            else:
                state = next_state

            if done:
                print(f"Episode {episode} finished after {episode_steps} steps with reward {reward}")
            
            p_loss = optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma)
            o_loss = optimize_model(opponent_memory, opponent_policy, opponent_target, opponent_optimizer, batch_size, gamma)
            
            if steps_done % target_update_freq == 0:
                print(f"Updating target networks at step {steps_done}")
                target_net.load_state_dict(policy_net.state_dict())
                opponent_target.load_state_dict(opponent_policy.state_dict())
                torch.save(policy_net.state_dict(), "latest.card.dqn.pt")
                test(episode)
            
            soft_update(policy_net, opponent_policy, tau=0.01)
            soft_update(target_net, opponent_target, tau=0.01)
            
            steps_done += 1
            episode_steps += 1
            
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

    torch.save(policy_net.state_dict(), "card.dqn.pt")
