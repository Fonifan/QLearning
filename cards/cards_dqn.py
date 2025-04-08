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
from card_env import Card, CardDurakEnv, Action
from versions.attention_complex_with_discard.attention_complex_with_discard import DQNComplexAttnDiscard as DQN, states_to_tensor, state_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 1000
MAX_HAND_SIZE = 20
MAX_TABLE_PAIRS = 6

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

def decode_state(state):
    normalized_deck_size = state["deck_size"]
    attacking = state["attacking"]
    
    hand = decode_card_int_list(state['hand'])
    table = decode_card_int_list(state['table'].flatten())
    
    numeric_state = ([normalized_deck_size, attacking])
    trump_tensor = decode_card_int_list([int(state["trump"])])
    discards = decode_card_int_list(state['discard'])
    print("deck_size:", normalized_deck_size)
    print("attacking:", attacking)
    print("hand:", hand)
    print("table:", table)
    print("numeric_state:", numeric_state)
    print("trump_tensor:", trump_tensor)
    print("discards:", discards)

def decode_card_int_list(card_int_list):
    card_list = []
    for card_int in card_int_list:
        card_list.append(Card.int_to_card(card_int))
    return card_list

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
