import random
from time import sleep
import torch
from cards_dqn import DQN, select_action
from card_env import CardDurakEnv, Action

MAX_HAND_SIZE = 20
MAX_TABLE_PAIRS = 6
input_dim = MAX_HAND_SIZE + MAX_TABLE_PAIRS * 2 + 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"Episode {episode}: Agent wins: {agent_wins}, Agent losses: {agent_losses}")
    print(f"Win rate: {agent_wins / total_games:.2%}")


if __name__ == "__main__":
    test(0)