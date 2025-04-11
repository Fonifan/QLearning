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
from versions.mlps.dqn_mlps import DQNMLPs as DQN, state_to_tensor, states_to_tensor
from plot_test import plot_results, plot_winrates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 1000
MAX_HAND_SIZE = 20
MAX_TABLE_PAIRS = 6
SELF_PLAY_START_EPISODE = 700
SELF_PLAY_END_EPISODE = 1700

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

def select_action_with_qs(state, valid_actions, epsilon, policy_net):
    if random.random() < epsilon:
        return random.choice(valid_actions)
    
    encoded = state_to_tensor(state).unsqueeze(0).to(device)
    q_values = policy_net(encoded)
    
    invalid_mask = torch.ones(q_values.size(), dtype=torch.bool)
    for action in valid_actions:
        invalid_mask[0][action] = False
    q_values[invalid_mask] = -1e9
    return torch.argmax(q_values).item(), q_values

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

def test(episode): #TODO seed env for test reproducibility
    env = CardDurakEnv()
    action_selectors = {
        "random": lambda _, valid_actions: random.choice(valid_actions),
        "policy_mlps": lambda state, valid_actions: select_action(state, valid_actions, epsilon=0, policy_net=policy_mlps)
    }
    for selector in action_selectors:
        print(f"Testing with action type: {selector}")
        agent_wins = 0
        agent_losses = 0
        total_games = 100
        for i in range(total_games):
            state = env.reset()
            done = False
            while not done:
                valid_actions = env._get_valid_actions(player_id=1)
                player_input = action_selectors[selector](state, valid_actions)

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
        write_results_to_csv(episode, selector, total_games, agent_wins, agent_losses)

def write_results_to_csv(episode, opponent, total_games, agent_wins, agent_losses):
    csv_file = "card_test_results.csv"
    win_rate = agent_wins / total_games if total_games > 0 else 0
    
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['Episode', 'Opponent', 'Total Games', 'Agent Wins', 'Agent Losses', 'Win Rate'])
        
        writer.writerow([episode, opponent, total_games, agent_wins, agent_losses, f"{win_rate:.4f}"])

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

def get_opponent_action(episode, state, valid_actions, oponent_net=None):
    # if episode >= 0 and episode <= 700:
    #     return random.choice(valid_actions)
    # elif episode > SELF_PLAY_START_EPISODE and episode <= SELF_PLAY_END_EPISODE:
    return select_action(state, valid_actions, epsilon=0, policy_net=oponent_net)
    # elif episode > 1700:
    #     return select_action(state, valid_actions, epsilon=0, policy_net=policy_mlps)

def write_winrate_to_csv(episode, wins, losses):
    csv_file = "card_winrate.csv"
    if wins + losses == 0:
        return
    win_rate = wins / (wins + losses)
    
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['Episode', 'Wins', 'Losses', 'Win Rate'])
        
        writer.writerow([episode, wins, losses, f"{win_rate:.4f}"])

if __name__ == "__main__":
    rewards_per_episode = []
    steps_done = 0
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    target_update_freq = 3000
    memory_size = 10000
    episodes = 2000

    env = CardDurakEnv()

    output_dim = env.action_space.n
    
    policy_net = DQN(output_dim).to(device)
    print(policy_net)
    print("Number of parameters in DQN:", sum(p.numel() for p in policy_net.parameters() if p.requires_grad))
    target_net = DQN(output_dim).to(device)

    policy_mlps = DQN(output_dim).to(device)
    policy_mlps.load_state_dict(torch.load("versions/mlps/best.pt", map_location=device))
    policy_mlps.eval()

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    memory = deque(maxlen=memory_size)
    opponent_memory = deque(maxlen=memory_size)
    wins = 0
    losses = 0
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
                opponent_action = get_opponent_action(episode, state, valid_actions, oponent_net=target_net)
                opponent_next_state, opponent_reward, opponent_done, truncated, _ = env.step(opponent_action, player_id=2)
                state = opponent_next_state
                if opponent_done:
                    done = True
                    reward = -opponent_reward  
            else:
                state = next_state

            if done:
                print(f"Episode {episode} finished after {episode_steps} steps with reward {reward}")
                if reward == 1:
                    wins += 1
                elif reward == -1:
                    losses += 1
            
            p_loss = optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma)
            
            if steps_done % target_update_freq == 0:
                write_winrate_to_csv(episode, wins, losses)
                wins = 0
                losses = 0
                print(f"Updating target networks at step {steps_done}")
                target_net.load_state_dict(policy_net.state_dict())
                torch.save(policy_net.state_dict(), "latest.card.dqn.pt")
                test(episode)
            
            steps_done += 1
            episode_steps += 1
            
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

    torch.save(policy_net.state_dict(), "card.dqn.pt")
    plot_results()
    plot_winrates()
