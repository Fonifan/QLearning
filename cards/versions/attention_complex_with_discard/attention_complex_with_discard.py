import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_HAND_SIZE = 20
MAX_TABLE_PAIRS = 6
MAX_DISCARDS = 36
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNComplexAttnDiscard(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNComplexAttnDiscard, self).__init__()
        self.hand_size = MAX_HAND_SIZE
        self.table_size = MAX_TABLE_PAIRS * 2 
        self.num_state_size = 2
        
        self.card_embedding = nn.Embedding(37, 16, padding_idx=0)
        
        self.num_state_fc = nn.Sequential(
            nn.Linear(self.num_state_size, 16),
            nn.ReLU()
        )
        
        self.hand_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
        self.table_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True)
        
        self.hand_fc = nn.Linear(32 * self.hand_size, 64)
        self.table_fc = nn.Linear(16 * self.table_size, 64)
        
        self.trump_fc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        self.discard_fc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        self.combine = nn.Linear(64 + 64 + 16 + 16 + 16, 128)
        self.hidden = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)
        
    def forward(self, x):

        pos = 0
        hand = x[:, pos:pos+self.hand_size].long()
        pos += self.hand_size
        
        table = x[:, pos:pos+self.table_size].long()
        pos += self.table_size
        
        numeric_state = x[:, pos:pos+2].float()
        pos += 2
        
        trump_int = x[:, pos].long()  # shape: [batch]
        pos += 1
        
        discards = x[:, pos:pos+MAX_DISCARDS].long()  # [batch, MAX_DISCARDS]
        
        hand_emb = self.card_embedding(hand)   # [batch, hand_size, 16]
        table_emb = self.card_embedding(table)   # [batch, table_size, 16]
        trump_emb = self.card_embedding(trump_int) # [batch, 16]
        discard_emb = self.card_embedding(discards)  # [batch, MAX_DISCARDS, 16]
        
        trump_expanded = trump_emb.unsqueeze(1)  # [batch, 1, 16]
        
        hand_key = torch.cat([hand_emb, trump_expanded], dim=1)  # [batch, hand_size+1, 16]
        hand_value = torch.cat([hand_emb, trump_expanded], dim=1)  # [batch, hand_size+1, 16]
        hand_att, _ = self.hand_attention(query=hand_emb, key=hand_key, value=hand_value)
        
        table_key = torch.cat([table_emb, trump_expanded], dim=1)  # [batch, table_size+1, 16]
        table_value = torch.cat([table_emb, trump_expanded], dim=1)  # [batch, table_size+1, 16]
        table_att, _ = self.table_attention(query=table_emb, key=table_key, value=table_value)
        
        hand_to_table, _ = self.cross_attention(query=hand_emb, key=table_key, value=table_value)
        
        hand_features = torch.cat([hand_att, hand_to_table], dim=2)  # [batch, hand_size, 32]
        hand_features = hand_features.flatten(start_dim=1)           # [batch, hand_size*32]
        table_features = table_att.flatten(start_dim=1)              # [batch, table_size*16]
        
        num_state_out = F.relu(self.num_state_fc(numeric_state))
        trump_out = F.relu(self.trump_fc(trump_emb))
        
        discard_summary = torch.mean(discard_emb, dim=1)  # [batch, 16]
        discard_out = F.relu(self.discard_fc(discard_summary))
        
        combined = torch.cat([
            F.relu(self.hand_fc(hand_features)),
            F.relu(self.table_fc(table_features)),
            num_state_out,
            trump_out,
            discard_out
        ], dim=1)
        
        x = F.relu(self.combine(combined))
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

def state_to_tensor(state):
    normalized_deck_size = state["deck_size"] / 36.0 # TODO magic number
    attacking = float(state["attacking"])
    
    hand = torch.IntTensor(state['hand'])
    table = torch.IntTensor(state['table'].flatten())
    
    numeric_state = torch.FloatTensor([normalized_deck_size, attacking])
    trump_tensor = torch.IntTensor([int(state["trump"])])
    discards = torch.IntTensor(state['discard'])
    
    return torch.cat([hand, table, numeric_state, trump_tensor, discards])

def states_to_tensor(states):
    tensors = []
    for state in states:
        tensors.append(state_to_tensor(state).to(device))
    return torch.stack(tensors)