
import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, max_trials, cost_threshold):
        super(DQN, self).__init__()
        self.max_trials = max_trials
        self.cost_threshold = cost_threshold
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


    def state_to_tensor(self, state):
        v = state['V']
        cost = state['cost'] / self.cost_threshold
        trials = state['trials'] / self.max_trials
        return torch.cat((torch.tensor([v], dtype=torch.float32),
                        torch.tensor([cost], dtype=torch.float32),
                        torch.tensor([trials], dtype=torch.float32),
        ))


    def states_to_tensor(self, states):
        tensors = []
        for state in states:
            tensors.append(self.state_to_tensor(state))
        return torch.stack(tensors)
