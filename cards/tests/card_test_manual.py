import random
from time import sleep
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from versions.attention_complex.dqn_complex_attention import DQNComplexAttn as DQN
from card_env import CardDurakEnv, Action
from cards_dqn import select_action

MAX_HAND_SIZE = 20
MAX_TABLE_PAIRS = 6
input_dim = MAX_HAND_SIZE + MAX_TABLE_PAIRS * 2 + 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    output_dim = 22  # env.action_space.n from card_env
    policy_net = DQN(input_dim, output_dim).to(device)
    try:
        policy_net.load_state_dict(torch.load("versions/attention_complex/best.pt", map_location=device))
        policy_net.eval()
        print(f"Trained model loaded successfully on {device}.")
    except Exception as e:
        print("Error loading model:", e)
        return

    env = CardDurakEnv()

    print("You are Player 1. Enter the corresponding number for your desired action when prompted.")
    agent_wins = 0
    agent_losses = 0
    total_games = 10
    for i in range(total_games):
        state = env.reset()
        done = False
        while not done:
            print("\n--- Your Turn ---")
            env.render(1)
            valid_actions = env._get_valid_actions(player_id=1)
            print("Valid actions:", valid_actions)
            try:
                player_input = int(input("Enter your chosen action: "))
            except ValueError:
                print("Invalid input. Please enter an integer corresponding to a valid action.")
                continue
            if player_input not in valid_actions:
                print("That action is not valid in the current state. Try again.")
                continue

            state, reward, done, _, _ = env.step(player_input, player_id=1)
            if done:
                print(f"Game over. You {'won' if reward > 0 else 'lost'} with reward: {reward}")
                agent_losses += 1 if reward > 0 else 0
                agent_wins += 1 if reward < 0 else 0
                break

            print("\n--- Opponent's Turn ---")
            env.render(2)
            valid_actions_opponent = env._get_valid_actions(player_id=2)
            opponent_action = select_action(state, valid_actions_opponent, epsilon=0, policy_net=policy_net)
            print("Opponent's valid actions:", valid_actions_opponent)
            print("Opponent selects action:", opponent_action)
            state, opp_reward, done, _, _ = env.step(opponent_action, player_id=2)
            if done:
                final_reward = -opp_reward
                print(f"Game over. You {'won' if final_reward > 0 else 'lost'} with reward: {final_reward}")
                agent_losses += 1 if reward > 0 else 0
                agent_wins += 1 if reward < 0 else 0
                break
    print(f"\nFinal Results: {total_games} games played.")
    print(f"Agent wins: {agent_wins}")
    print(f"Agent losses: {agent_losses}")
    print(f"Win rate: {agent_wins / total_games:.2%}")

if __name__ == "__main__":
    main()