import torch
from dqn import DQN
from patient import Actions, Patient
from dqn_train import select_action

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


env = Patient(V, ALPHA_X, COST_X, COST_THRESHOLD, max_trials=MAX_TRIALS)
output_dim = env.action_space.n
input_dim = env.observation_space.shape[0]
policy_net = DQN(input_dim, output_dim, MAX_TRIALS, COST_THRESHOLD)
policy_net.load_state_dict(torch.load("rw.pt"))
policy_net.eval()
test_episodes = 1
for episode in range(test_episodes):
    state, info = env.reset()
    done = False
    while not done:
        env.render()
        action = select_action(state, 0, policy_net, env)
        print("Action:", Actions.to_str(action))
        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            done = True
            env.render()
        state = next_state
