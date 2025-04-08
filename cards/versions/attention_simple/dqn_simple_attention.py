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

class DQNSimpleAttn(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNSimpleAttn, self).__init__()
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