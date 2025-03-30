import gymnasium as gym
import torch
import minigrid
from dqn import DQN, select_action, NUM_ACTIONS, ENVIRONMENT_ID

env = gym.make(ENVIRONMENT_ID, render_mode="human")
policy_net = DQN(NUM_ACTIONS)
policy_net.load_state_dict(torch.load('dqn.pt', weights_only=True))
policy_net.eval()
for i in range(100):
    terminated = False
    e_steps = 0
    state, info = env.reset(seed=42)
    while not terminated:
        action = select_action(state, 0, policy_net)
        print("Step:", e_steps, "Action:", action)
        state, reward, terminated, _, _ = env.step(action)
        if terminated:
            print("Terminated")
            break
        e_steps += 1
    print(f"Episode {i} finished after {e_steps} steps")
env.close()
