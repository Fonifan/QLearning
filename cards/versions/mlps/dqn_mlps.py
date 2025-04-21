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
        
    def forward(self, hand, table, trump, deck_size, is_attack, discards):
        hand_emb = self.card_embedding(hand)   # [batch, hand_size, 16]
        table_emb = self.card_embedding(table)   # [batch, table_size, 16]
        trump_emb = self.card_embedding(trump) # [batch, 16]
        discard_emb = self.card_embedding(discards)  # [batch, MAX_DISCARDS, 16]

        hand_emb_flat = hand_emb.view(hand_emb.size(0), -1)    # [batch, 16 * MAX_HAND_SIZE]
        table_emb_flat = table_emb.view(table_emb.size(0), -1) # [batch, 16 * MAX_TABLE_SIZE]
        discard_emb_flat = discard_emb.view(discard_emb.size(0), -1)  # [batch, 16 * MAX_DISCARDS]
        trump_emb_flat = trump_emb.view(trump_emb.size(0), -1)

        hand_out = F.relu(self.fc_hand1(hand_emb_flat))
        hand_out = F.relu(self.fc_hand2(torch.cat([hand_out, trump_emb_flat], dim=1)))

        table_out = F.relu(self.fc_table1(table_emb_flat))
        table_out = F.relu(self.fc_table2(torch.cat([table_out, trump_emb_flat], dim=1)))

        state_out = F.relu(self.fc_state(torch.cat([deck_size, is_attack], dim=1)))

        discard_out = F.relu(self.fc_discard(discard_emb_flat))
    
        combined = torch.cat([hand_out, table_out, discard_out, state_out], dim=1)
        
        x = F.relu(self.combine(combined))
        x = F.relu(self.hidden(x))
        x = self.out(x)
        
        return x

def states_to_tensor(states):
    tensors = [state_to_tensor(s) for s in states]
    batch = {}
    for key in tensors[0]:
        batch[key] = torch.stack([t[key] for t in tensors])
    return batch

def state_to_tensor(state):
    hand = torch.tensor(state['hand'], dtype=torch.long)
    table = torch.tensor(state['table'].flatten() if hasattr(state['table'], 'flatten') else state['table'],
                         dtype=torch.long)
    deck_size = torch.tensor([state['deck_size'] / 36.0], dtype=torch.float)
    is_attack = torch.tensor([state['attacking']], dtype=torch.float)
    trump = torch.tensor([int(state['trump'])], dtype=torch.long)
    discards = torch.tensor(state['discard'], dtype=torch.long)

    return {
        'hand': hand,
        'table': table,
        'deck_size': deck_size,
        'is_attack': is_attack,
        'trump': trump,
        'discards': discards
    }