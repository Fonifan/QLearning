
from patient import Patient
import torch.nn as nn
import torch
import random
from collections import deque
import torch.optim as optim
from dqn import DQN
import matplotlib.pyplot as plt

MAX_TRIALS = 20
V = 3
ALPHA_X = {
    0: 0.1,
    1: 0.2,
    2: 0.4
}
COST_X = {
    0: 0.15,
    1: 0.3,
    2: 0.7
}
COST_THRESHOLD = 5


def select_action(state, epsilon, model, env):
    if random.random() < epsilon:
        return random.randrange(env.action_space.n)
    else:
        state_tensor = model.state_to_tensor(state)
        q_values = model(state_tensor)
        return torch.argmax(q_values).item()


if __name__ == "__main__":

    steps_done = 0
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.99
    batch_size = 64
    target_update_freq = 2000
    memory_size = 10000
    episodes = 1000

    env = Patient(V, ALPHA_X, COST_X, COST_THRESHOLD, max_trials=MAX_TRIALS)

    output_dim = env.action_space.n
    input_dim = env.observation_space.shape[0]

    policy_net = DQN(input_dim, output_dim, MAX_TRIALS, COST_THRESHOLD)
    target_net = DQN(input_dim, output_dim, MAX_TRIALS, COST_THRESHOLD)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size)
    episode_rewards = []
    episode_costs = []

    for episode in range(episodes):
        state, info = env.reset()
        done = False
        episode_steps = 0
        cumulative_reward = 0
        while not done:
            action = select_action(state, epsilon, policy_net, env)
            next_state, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            memory.append((state, action, reward, next_state, done))

            if terminated or truncated:
                done = True
                print(
                    f"Episode {episode} finished after {next_state['trials']} steps with cumulative reward {cumulative_reward}, cost {info['cost']}")
                episode_rewards.append(cumulative_reward)
                episode_costs.append(info['cost'])

            state = next_state

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
                    *batch)

                state_batch = policy_net.states_to_tensor(state_batch)
                action_batch = torch.LongTensor(action_batch).unsqueeze(1)
                reward_batch = torch.FloatTensor(reward_batch)
                next_state_batch = policy_net.states_to_tensor(
                    next_state_batch)
                done_batch = torch.IntTensor(done_batch)

                q_values = policy_net(state_batch).gather(
                    1, action_batch).squeeze()

                with torch.no_grad():
                    max_next_q_values = target_net(next_state_batch).max(1)[0]
                    target_q_values = reward_batch + gamma * \
                        max_next_q_values * (1 - done_batch)

                loss = nn.MSELoss()(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % target_update_freq == 0:
                print(f"Updating target network at step {steps_done}")
                target_net.load_state_dict(policy_net.state_dict())

            steps_done += 1
            episode_steps += 1

        epsilon = max(epsilon_min, epsilon_decay * epsilon)

    torch.save(policy_net.state_dict(), "rw.pt")
    
    print("Training completed and model saved.")

    plt.figure()
    plt.plot(episode_rewards, color='tab:blue')
    plt.grid(visible=True)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Episode Rewards over Time')
    plt.savefig('rewards.png')

    # Plot and save costs
    plt.figure()
    plt.plot(episode_costs, color='tab:red')
    plt.grid(visible=True)
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.title('Episode Costs over Time')
    plt.savefig('costs.png')

    plt.show()
