import sys
import numpy as np
import gymnasium as gym

# Simplifications for the card game "Durak"
# * Player can play one card per turn
# * First player always starts

class Action:
    TAKE = 0
    DISCARD = 1
    PLAY_1 = 2
    PLAY_2 = 3
    PLAY_3 = 4
    PLAY_4 = 5
    PLAY_5 = 6
    PLAY_6 = 7
    PLAY_7 = 8
    PLAY_8 = 9
    PLAY_9 = 10
    PLAY_10 = 11
    PLAY_11 = 12
    PLAY_12 = 13
    PLAY_13 = 14
    PLAY_14 = 15
    PLAY_15 = 16
    PLAY_16 = 17
    PLAY_17 = 18
    PLAY_18 = 19
    PLAY_19 = 20
    PLAY_20 = 21

class Card:
    suits = ["hearts", "diamonds", "clubs", "spades"]
    pretty_suit = {
        "hearts": "♥",
        "diamonds": "♦",
        "clubs": "♣",
        "spades": "♠"
    }
    ranks = ["6", "7", "8", "9", "10", "J", "Q", "K", "A"]

    rank_to_int = {rank: i for i, rank in enumerate(ranks)}
    suit_to_int = {suit: i for i, suit in enumerate(suits)}
    NULL_CARD = 0
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        if self.rank == "NULL" or self.suit == "NULL":
            return "NULL"
    
        return f"{self.rank}{self.pretty_suit[self.suit]}"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    def to_int(self):
        return Card.rank_to_int[self.rank] * len(Card.suits) + Card.suit_to_int[self.suit] + 1  # +1 to avoid NULL_CARD being 0
    
    def int_to_card(card_int: int):
        if card_int == Card.NULL_CARD:
            return Card("NULL", "NULL")
        rank = Card.ranks[(card_int - 1) // len(Card.suits)]
        suit = Card.suits[(card_int - 1) % len(Card.suits)]
        return Card(rank, suit)

class Player:
    def __init__(self, name: str):
        self.name = name
        self.hand = []

    def __repr__(self):
        return f"Player(name={self.name}, hand={self.hand})"
    
    def play_card(self, card: Card):
        if card in self.hand:
            self.hand.remove(card)
        else:
            raise ValueError(f"Card {card} not in hand")
        
    def add_card(self, card: Card):
        self.hand.append(card)
    
    def add_cards(self, cards: list):
        self.hand.extend(cards)

class GamePhase:
    ATTACK = "Attack"
    DEFENSE = "Defense"

class CardDurakEnv(gym.Env):

    def __init__(self, num_cards: int = 36):
        self.num_cards = num_cards
        self.play_action_offset = 2
        self.phase = GamePhase.ATTACK
        self._action_idx_to_action = {
            0: Action.TAKE,  # take
            1: Action.DISCARD,  # discard
            2: Action.PLAY_1,  # play 1st card
            3: Action.PLAY_2,  # play 2nd card
            4: Action.PLAY_3,  # play 3rd card
            5: Action.PLAY_4,  # play 4th card
            6: Action.PLAY_5,  # play 5th card
            7: Action.PLAY_6,  # play 6th card 
            8: Action.PLAY_7,  # play 7th card
            9: Action.PLAY_8,  # play 8th card
            10: Action.PLAY_9,  # play 9th card
            11: Action.PLAY_10,  # play 10th card
            12: Action.PLAY_11,  # play 11th card
            13: Action.PLAY_12,  # play 12th card
            14: Action.PLAY_13,  # play 13th card
            15: Action.PLAY_14,  # play 14th card
            16: Action.PLAY_15,  # play 15th card
            17: Action.PLAY_16,  # play 16th card
            18: Action.PLAY_17,  # play 17th card
            19: Action.PLAY_18,  # play 18th card
            20: Action.PLAY_19,  # play 19th card
            21: Action.PLAY_20  # play 20th card
        }
        self.MAX_PLAYABLE_CARD = len(self._action_idx_to_action) - self.play_action_offset
        self.action_space = gym.spaces.Discrete(len(self._action_idx_to_action))
        
        self.observation_space = gym.spaces.Dict({
            "hand": gym.spaces.Box(low=Card.NULL_CARD, high=self.MAX_PLAYABLE_CARD, shape=(self.MAX_PLAYABLE_CARD,), dtype=np.int32),  # List of card integers
            "deck_size": gym.spaces.Discrete(num_cards + 1),  # Integer
            "table": gym.spaces.Box(low=Card.NULL_CARD, high=num_cards-1, shape=(6, 2), dtype=np.int32),  # Nested list structure
            "discard": gym.spaces.Box(low=Card.NULL_CARD, high=num_cards-1, shape=(num_cards,), dtype=np.int32),  # List of card integers
            "trump": gym.spaces.Discrete(num_cards),  # Integer
            "attacking": gym.spaces.Discrete(2)  # Boolean converted to int
        })

    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self.deck = self.generate_deck(self.num_cards)
        self.player_1 = Player("Player 1")
        self.player_2 = Player("Player 2")
        self.id_to_player = {
            1: self.player_1,
            2: self.player_2
        }
        self.table = []
        self.discarded = []
        self.player_1.add_cards(self.get_from_deck(6))
        self.player_2.add_cards(self.get_from_deck(6))
        self.attacking_player = self.player_1
        self.trump = self.deck[-1]
        self.turns = 0
        self.phase = GamePhase.ATTACK

        return self._get_obs(1)

    def _get_obs(self, player_id: int = 1):
        player = self.id_to_player[player_id]
        
        hand_transformed = self._transform_card_list(player.hand)
        hand_padded = hand_transformed + [Card.NULL_CARD] * (self.MAX_PLAYABLE_CARD - len(hand_transformed))
        hand_padded = hand_padded[:self.MAX_PLAYABLE_CARD] 


        table_transformed = self._transform_table(self.table)
        table_padded = []
        for i in range(6):
            if i < len(table_transformed):
                row = table_transformed[i] + [Card.NULL_CARD] * (2 - len(table_transformed[i]))
                row = row[:2] 
            else:
                row = [Card.NULL_CARD, Card.NULL_CARD]
            table_padded.append(row)
        
        discard_transformed = self._transform_card_list(self.discarded)
        discard_padded = discard_transformed + [Card.NULL_CARD] * (self.num_cards - len(discard_transformed))
        discard_padded = discard_padded[:self.num_cards] 
        
        return {
            "hand": np.array(hand_padded, dtype=np.int32),
            "deck_size": len(self.deck),
            "table": np.array(table_padded, dtype=np.int32),
            "discard": np.array(discard_padded, dtype=np.int32),
            "trump": self.trump.to_int(),
            "attacking": 1 if self.attacking_player == player else 0,  # Convert boolean to int
        }
    
    def _transform_card_list(self, cards: list[Card]):
        return [card.to_int() for card in cards]
    
    def _transform_table(self, table: list[list[Card]]):
        return [self._transform_card_list(cards) for cards in table]
    
    def _get_valid_actions(self, player_id: int):
        valid_actions = []
        player = self.id_to_player[player_id]
        is_attacking = self.attacking_player == player
        if len(self.table) > 0 and  not is_attacking:
            valid_actions.append(Action.TAKE)
        if len(self.table) > 0 and is_attacking:
            valid_actions.append(Action.DISCARD)

        valid_actions.extend(self._get_valid_play_actions(player.hand, is_attacking))
        return valid_actions

    def _get_valid_play_actions(self, hand: list[Card], is_attacking: bool):
        limited_hand = hand[:self.MAX_PLAYABLE_CARD]
        if is_attacking:
            if not self.can_throw_in():
                return []
            if len(self.table) == 0:
                return [i + self.play_action_offset for i in range(len(limited_hand))]
            valid = set()
            for table_card_tuple in self.table:
                for card in table_card_tuple:
                    for i, hand_card in enumerate(limited_hand):
                        if hand_card.rank == card.rank:
                            valid.add(i + self.play_action_offset)
        else:
            card_to_beat = self.table[-1][0]
            valid = set()
            for i, card in enumerate(limited_hand):
                if ((card.suit == card_to_beat.suit and Card.rank_to_int[card.rank] > Card.rank_to_int[card_to_beat.rank]) or 
                    (card.suit == self.trump.suit and card_to_beat.suit != self.trump.suit)):
                    valid.add(i + self.play_action_offset)
        return list(valid)

    def _get_defending_player(self):
        if self.attacking_player == self.player_1:
            return self.player_2
        else:
            return self.player_1
      
    def can_throw_in(self):
        max_table = 5 if self.turns == 0 else 6
        return len(self.table) <= max_table and len(self._get_defending_player().hand) > 0

    def step(self, action, player_id: int):
        if action == Action.TAKE:
            self.take(player_id)
            is_finished = self.is_finished()
        elif action == Action.DISCARD:
            self.discard(player_id)
            is_finished = self.is_finished()
        elif action >= self.play_action_offset:
            self.play_card(player_id, action - self.play_action_offset)
            is_finished = False

        other_player_id = 2 if player_id == 1 else 1
        if is_finished != False:
            return self._get_obs(other_player_id), 1 if is_finished == player_id else -1, True, False, {}
        
        return self._get_obs(other_player_id), 0, False, False, {}
    
    def render(self, player_id, mode="ansi"):
        """Render the current game state as text."""
        if mode != "ansi":
            raise NotImplementedError(f"Render mode {mode} is not supported.")
            
        output = []
        
        # Game state header
        output.append(f"Phase: {self.phase}")
        output.append(f"Attacking player: {'Player 1' if self.attacking_player == self.player_1 else 'Player 2'}")
        output.append(f"Trump: {str(self.trump)}")
        output.append(f"Deck remaining: {len(self.deck)} cards")
        output.append(f"Discard pile: {len(self.discarded)} cards")
        
        # Table cards
        output.append("\n==== TABLE ====")
        if not self.table:
            output.append("Empty")
        else:
            for i, card_pair in enumerate(self.table):
                if len(card_pair) == 1:
                    output.append(f"{i+1}. {str(card_pair[0])}")
                else:
                    output.append(f"{i+1}. {str(card_pair[0])}/"
                                f"{str(card_pair[1])}")
        
        output.append("\n==== PLAYER HAND ====")
        player = self.id_to_player[player_id]
        if not player.hand:
            output.append("Empty")
        else:
            cards = [f"\n{i+self.play_action_offset}: {str(card)}" for i, card in enumerate(player.hand)]
            output.append("".join(cards))
        
        result = "\n".join(output)
        print(result)
        return result

    def take(self, player_id: int):
        player = self.id_to_player[player_id]
        player.add_cards(self.flatten_table(self.table))
        self.end_turn(Action.TAKE, player_id)

    def discard(self, player_id: int):
        self.discarded.extend(self.flatten_table(self.table))
        self.end_turn(Action.DISCARD, player_id)

    def play_card(self, player_id: int, card_index: int):
        if self.phase == GamePhase.DEFENSE:
            player = self.id_to_player[player_id]
            card = player.hand[card_index]
            player.play_card(card)
            self.table[-1].append(card)
            self.phase = GamePhase.ATTACK
        else:
            player = self.id_to_player[player_id]
            card = player.hand[card_index]
            player.play_card(card)
            self.table.append([card])
            self.phase = GamePhase.DEFENSE
            
    def flatten_table(self, table: list[list[Card]]):
        return [card for sublist in table for card in sublist]

    def get_other_player(self, player_id: int):
        if player_id == 1:
            return self.player_2
        else:
            return self.player_1

    def end_turn(self, action: Action, player_id: int):
        if action == Action.DISCARD:
            self.attacking_player = self._get_defending_player()
        
        if len(self.player_1.hand) < 6:
            self.player_1.add_cards(self.get_from_deck(6 - len(self.player_1.hand)))
        if len(self.player_2.hand) < 6:
            self.player_2.add_cards(self.get_from_deck(6 - len(self.player_2.hand)))
        self.turns += 1
        self.table = []
        self.phase = GamePhase.ATTACK

    def is_finished(self):
        if len(self.player_1.hand) == 0:
            return 1
        elif len(self.player_2.hand) == 0:
            return 2
        
        return False

    def generate_deck(self, num_cards: int):
        deck = []
        for suit in Card.suits:
            for rank in Card.ranks:
                deck.append(Card(rank, suit))
        np.random.shuffle(deck)
        return deck[:num_cards]

    def get_from_deck(self, num: int):
        cards = self.deck[:num] # TODO check if deck is empty
        self.deck = self.deck[num:]
        return cards