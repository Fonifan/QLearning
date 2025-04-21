import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 1000
MAX_HAND_SIZE = 20
MAX_TABLE_PAIRS = 6
MAX_TABLE_SIZE = MAX_TABLE_PAIRS * 2
MAX_DISCARDS = 36
OPPONENT_OUT = 32
MAIN_OUTPUT = 64
class OpponentModel(nn.Module):
    def __init__(self, output_dim):
        super(OpponentModel, self).__init__()
        self.fc_table1 = nn.Linear(16 * MAX_TABLE_SIZE, 128)
        self.fc_table2 = nn.Linear(128 + 16, 64)
        
        self.fc_discard = nn.Linear(16 * MAX_DISCARDS, 32)

        self.fc_state = nn.Linear(4, 16)
        
        self.combine = nn.Linear(64 + 32 + 16, 128)
        self.out = nn.Linear(128, output_dim)
        
    def forward(self, table, deck_size, attacking, hand_size, trump, discard, action):
        table_out = F.relu(self.fc_table1(table))
        table_out = F.relu(self.fc_table2(torch.cat([table_out, trump], dim=1)))

        meta_state = torch.cat([deck_size, attacking, hand_size, action], dim=1)
        state_out = F.relu(self.fc_state(meta_state))

        discard_out = F.relu(self.fc_discard(discard))
    
        combined = torch.cat([table_out, discard_out, state_out], dim=1)
        
        x = F.relu(self.combine(combined))
        x = self.out(x)
        return x
    
class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()

        self.fc_hand1 = nn.Linear(16 * MAX_HAND_SIZE, 128)
        self.fc_hand2 = nn.Linear(128 + 16, 64)
        
        self.fc_table1 = nn.Linear(16 * MAX_TABLE_SIZE, 128)
        self.fc_table2 = nn.Linear(128 + 16, 64)
        
        self.fc_state = nn.Linear(2, 8)
        
        self.fc_discard = nn.Linear(16 * MAX_DISCARDS, 32)
        self.combine = nn.Linear(64 + 64 + 32 + 8, 128)
        self.out = nn.Linear(128, output_dim)
    
    def forward(self, hand, table, deck_size, is_attacking, trump, discards):
        hand_out = F.relu(self.fc_hand1(hand))
        hand_out = F.relu(self.fc_hand2(torch.cat([hand_out, trump], dim=1)))

        table_out = F.relu(self.fc_table1(table))
        table_out = F.relu(self.fc_table2(torch.cat([table_out, trump], dim=1)))

        meta_state = torch.cat([deck_size, is_attacking], dim=1)
        state_out = F.relu(self.fc_state(meta_state))

        discard_out = F.relu(self.fc_discard(discards))
    
        combined = torch.cat([hand_out, table_out, discard_out, state_out], dim=1)
        
        x = F.relu(self.combine(combined))
        x = self.out(x)
        return x

class DRON(nn.Module):
    def __init__(self, output_dim):
        super(DRON, self).__init__()
        self.card_embedding = nn.Embedding(37, 16, padding_idx=0)
        self.main = DQN(MAIN_OUTPUT)
        self.opponent_model = OpponentModel(OPPONENT_OUT)
        self.hidden = nn.Linear(MAIN_OUTPUT + OPPONENT_OUT, 64)
        self.out = nn.Linear(64, output_dim)
        
    def forward(self, hand, table, deck_size, is_attack, trump, discards,
                opp_table, opp_deck_size, opp_attacking, opp_hand_size, opp_trump, opp_discard, opp_action):

        hand_emb = self.card_embedding(hand)           # [batch, MAX_HAND_SIZE, 16]
        table_emb = self.card_embedding(table)           # [batch, MAX_TABLE_SIZE, 16]
        trump_emb = self.card_embedding(trump)           # [batch, 1, 16]
        discard_emb = self.card_embedding(discards)        # [batch, MAX_DISCARDS, 16]

        hand_emb_flat = hand_emb.view(hand_emb.size(0), -1)
        table_emb_flat = table_emb.view(table_emb.size(0), -1)
        discard_emb_flat = discard_emb.view(discard_emb.size(0), -1)
        trump_emb_flat = trump_emb.view(trump_emb.size(0), -1)

        main_out = self.main(hand_emb_flat, table_emb_flat, deck_size, is_attack, trump_emb_flat, discard_emb_flat)

        opp_table_emb = self.card_embedding(opp_table)
        opp_discard_emb = self.card_embedding(opp_discard)
        opp_table_emb_flat = opp_table_emb.view(opp_table_emb.size(0), -1)
        opp_discard_emb_flat = opp_discard_emb.view(opp_discard_emb.size(0), -1)

        opponent_out = self.opponent_model(
            opp_table_emb_flat, opp_deck_size, opp_attacking, opp_hand_size,
            trump_emb_flat, opp_discard_emb_flat, opp_action)

        x = torch.cat([main_out, opponent_out], dim=1)
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

def state_to_tensor(state):
    hand = torch.tensor(state['hand'], dtype=torch.long)
    table = torch.tensor(state['table'].flatten(), dtype=torch.long)
    deck_size = torch.tensor([state['deck_size'] / 36.0], dtype=torch.float)
    is_attack = torch.tensor([state['attacking']], dtype=torch.float)
    trump = torch.tensor([int(state['trump'])], dtype=torch.long)
    discards = torch.tensor(state['discard'], dtype=torch.long)
    if "opp_deck_size" in state:
        opp_table = torch.tensor(state['opp_table'].flatten() if hasattr(state['opp_table'], 'flatten') else state['opp_table'], dtype=torch.long)
        opp_deck_size = torch.tensor([state['opp_deck_size'] / 36.0], dtype=torch.float)
        opp_attacking = torch.tensor([state['opp_attacking']], dtype=torch.float)
        opp_hand_size = torch.tensor([state['opp_hand_size']], dtype=torch.float)
        opp_trump = torch.tensor([int(state['opp_trump'])], dtype=torch.long)
        opp_discard = torch.tensor(state['opp_discard'], dtype=torch.long)
        opp_action = torch.tensor([int(state['opp_action'])], dtype=torch.long)
        return {
            'hand': hand,
            'table': table,
            'deck_size': deck_size,
            'is_attack': is_attack,
            'trump': trump,
            'discards': discards,
            'opp_table': opp_table,
            'opp_deck_size': opp_deck_size,
            'opp_attacking': opp_attacking,
            'opp_hand_size': opp_hand_size,
            'opp_trump': opp_trump,
            'opp_discard': opp_discard,
            'opp_action': opp_action
        }
    else:
        zeros_table = torch.zeros(table.size(), dtype=torch.long)
        zeros_discards = torch.zeros_like(discards)
        return {
            'hand': hand,
            'table': table,
            'deck_size': deck_size,
            'is_attack': is_attack,
            'trump': trump,
            'discards': discards,
            'opp_table': zeros_table,
            'opp_deck_size': torch.tensor([0.0]),
            'opp_attacking': torch.tensor([0.0]),
            'opp_hand_size': torch.tensor([0.0]),
            'opp_trump': torch.tensor([0], dtype=torch.long),
            'opp_discard': zeros_discards,
            'opp_action': torch.tensor([0], dtype=torch.long)
        }

def states_to_tensor(states):
    tensors = [state_to_tensor(s) for s in states]
    batch = {}
    for key in tensors[0]:
        batch[key] = torch.stack([t[key] for t in tensors])
    return batch

