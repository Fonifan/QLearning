import random
from time import sleep
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from card_env import CardDurakEnv, Action
from cards_dqn import select_action
from versions.mlps.dqn_mlps import DQNMLPs as DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    output_dim = 22 
    policy_net1 = DQN(output_dim).to(device)
    try:
        policy_net1.load_state_dict(torch.load("versions/mlps_stages/ex1/best.pt", map_location=device))
        policy_net1.eval()
        print(f"Player 1 (Complex Attention) model loaded successfully on {device}.")
    except Exception as e:
        print("Error loading Player 1 model:", e)
        return

    policy_net2 = DQN(output_dim).to(device)
    try:
        policy_net2.load_state_dict(torch.load("versions/mlps_stages/ex2/best.pt", map_location=device))
        policy_net2.eval()
        print(f"Player 2 (Simple Attention) model loaded successfully on {device}.")
    except Exception as e:
        print("Error loading Player 2 model:", e)
        return

    env = CardDurakEnv()

    wins_player1 = 0
    wins_player2 = 0
    total_games = 500

    for i in range(total_games):
        state = env.reset()
        done = False
        current_player = 1
        while not done:
            if current_player == 1:
                valid_actions = env._get_valid_actions(player_id=1)
                action = select_action(state, valid_actions, epsilon=0, policy_net=policy_net1)
                state, reward, done, _, _ = env.step(action, player_id=1)
                if done:
                    if reward > 0:
                        wins_player1 += 1
                        print(f"Player 1 wins! Game {i+1}/{total_games}")
                    else:
                        wins_player2 += 1
                        print(f"Player 2 wins! Game {i+1}/{total_games}")
                    break
                current_player = 2
            else:
                valid_actions = env._get_valid_actions(player_id=2)
                action = select_action(state, valid_actions, epsilon=0, policy_net=policy_net2)
                state, reward, done, _, _ = env.step(action, player_id=2)
                if done:
                    if reward > 0:
                        wins_player2 += 1
                        print(f"Player 2 wins! Game {i+1}/{total_games}")
                    else:
                        wins_player1 += 1
                        print(f"Player 1 wins! Game {i+1}/{total_games}")
                    break
                current_player = 1

    print(f"\nFinal Results: {total_games} games played.")
    print(f"Player 1 (MLPS) wins: {wins_player1}")
    print(f"Player 2 (MLPS with stages) wins: {wins_player2}")
    print(f"Win rates - Player 1: {wins_player1/total_games:.2%}, Player 2: {wins_player2/total_games:.2%}")

if __name__ == "__main__":
    main()