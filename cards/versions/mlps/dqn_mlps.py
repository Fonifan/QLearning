import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 1000
MAX_HAND_SIZE = 20
MAX_TABLE_PAIRS = 6
MAX_TABLE_SIZE = MAX_TABLE_PAIRS * 2
MAX_DISCARDS = 36

class DQNMLPs(nn.Module):
    def __init__(self, output_dim):
        super(DQNMLPs, self).__init__()
        
        self.card_embedding = nn.Embedding(37, 16, padding_idx=0) 
        self.fc_hand1 = nn.Linear(16 * MAX_HAND_SIZE, 128)
        self.fc_hand2 = nn.Linear(128 + 16, 64) # + 1 for trump
        
        self.fc_table1 = nn.Linear(16 * MAX_TABLE_SIZE, 128)
        self.fc_table2 = nn.Linear(128 + 16, 64) # + 1 for trump
        
        self.fc_state = nn.Linear(2, 8)
        
        self.fc_discard = nn.Linear(16 * MAX_DISCARDS, 32)
        self.combine = nn.Linear(64 + 64 + 32 + 8, 128)
        self.hidden = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)
        
    def forward(self, x):
        pos = 0
        hand = x[:, pos:pos+MAX_HAND_SIZE].long()
        pos += MAX_HAND_SIZE
        
        table = x[:, pos:pos+MAX_TABLE_SIZE].long()
        pos += MAX_TABLE_SIZE
        
        numeric_state = x[:, pos:pos+2].float()
        pos += 2
        
        trump_int = x[:, pos].long()  # shape: [batch]
        pos += 1
        
        discards = x[:, pos:pos+MAX_DISCARDS].long()  # [batch, MAX_DISCARDS]
        
        hand_emb = self.card_embedding(hand)   # [batch, hand_size, 16]
        table_emb = self.card_embedding(table)   # [batch, table_size, 16]
        trump_emb = self.card_embedding(trump_int) # [batch, 16]
        discard_emb = self.card_embedding(discards)  # [batch, MAX_DISCARDS, 16]

        hand_emb_flat = hand_emb.view(hand_emb.size(0), -1)    # [batch, 16 * MAX_HAND_SIZE]
        table_emb_flat = table_emb.view(table_emb.size(0), -1) # [batch, 16 * MAX_TABLE_SIZE]
        discard_emb_flat = discard_emb.view(discard_emb.size(0), -1)  # [batch, 16 * MAX_DISCARDS]

        hand_out = F.relu(self.fc_hand1(hand_emb_flat))
        hand_out = F.relu(self.fc_hand2(torch.cat([hand_out, trump_emb], dim=1)))

        table_out = F.relu(self.fc_table1(table_emb_flat))
        table_out = F.relu(self.fc_table2(torch.cat([table_out, trump_emb], dim=1)))

        state_out = F.relu(self.fc_state(numeric_state))

        discard_out = F.relu(self.fc_discard(discard_emb_flat))
    
        combined = torch.cat([hand_out, table_out, discard_out, state_out], dim=1)
        
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
    normalized_deck_size = state["deck_size"] / 36.0 # TODO magic number
    attacking = float(state["attacking"])
    
    hand = torch.IntTensor(state['hand'])
    table = torch.IntTensor(state['table'].flatten())
    
    numeric_state = torch.FloatTensor([normalized_deck_size, attacking])
    trump_tensor = torch.IntTensor([int(state["trump"])])
    discards = torch.IntTensor(state['discard'])
    
    return torch.cat([hand, table, numeric_state, trump_tensor, discards])