import sys
import copy

from nn_utils import SCORE_SCALAR
from rules import is_card_valid, is_score_card
from nn_utils import print_a_memory

from game import Game


class IntelligentGame(Game):
    def __init__(self, players, buffer_size=2**15, simulation_time_limit=1, verbose=False, out_file=sys.stdout):
        super(IntelligentGame, self).__init__(players, verbose, out_file=out_file)

        self.simulation_time_limit = simulation_time_limit


    def reset(self):
        super(IntelligentGame, self).reset()

        self._short_memory = []


    def get_memory(self):
        return self._short_memory


    def step(self, played_card=None):
        hand_cards = self._player_hands[self.current_player_idx]

        if played_card is None:
            played_card, results = self.players[self.current_player_idx].play_card(self, simulation_time_limit=self.simulation_time_limit)

        if not is_card_valid(hand_cards, self.trick, played_card, self.trick_nr, self.is_heart_broken):
            raise ValueError('{} round - Player {} ({}) played an invalid card {}({}) to the trick {}.'.format(\
                self.trick_nr, self.current_player_idx, type(self.players[self.current_player_idx]).__name__, played_card, hand_cards, self.trick))

        if played_card not in self._player_hands[self.current_player_idx]:
            raise ValueError("{} round - Not found {} card in this Player-{} hand cards({})".format(\
                self.trick_nr, played_card, self.current_player_idx, self._player_hands[self.current_player_idx]))

        valid_cards, probs = [], []
        for card, prob in results:
            valid_cards.append(card)
            probs.append(prob)

        must_cards = [[], [], [], []]
        for player_idx, cards in self.players[self.current_player_idx].transfer_cards.items():
            must_cards[player_idx] = cards

        void_info, score_cards = [[], [], [], []], [[], [], [], []]
        for player_idx, cards in enumerate(self._cards_taken):
            for card in sorted(cards):
                if is_score_card(card):
                    score_cards[player_idx].append(card)

            for sub_player_idx, info in self.players[player_idx].void_info.items():
                for suit, is_void in sorted(info.items(), key=lambda x: x[0]):
                    void_info[player_idx].append(is_void)

        self.player_action_pos[self.current_player_idx][self.trick_nr] = len(self.trick)+1

        remaining_cards = self.players[0].get_remaining_cards(hand_cards)
        expose_info = [2 if player.expose else 1 for player in self.players]

        self._short_memory.append([remaining_cards[:],
                                  self.trick[:],
                                  must_cards,
                                  self._historical_cards,
                                  score_cards,
                                  hand_cards[:],
                                  valid_cards,
                                  expose_info,
                                  void_info,
                                  probs,
                                  self.trick_nr,
                                  self.player_action_pos,
                                  self.player_winning_info,
                                  self.current_player_idx])

        self._historical_cards[self.current_player_idx].append(played_card)
        self._player_hands[self.current_player_idx].remove(played_card)
        self.trick.append(played_card)

        for i in range(4):
            self.players[i].see_played_trick(played_card, self)

        self.current_player_idx = (self.current_player_idx+1)%4
        if len(self.trick) == 4:
            self.round_over()


    def round_over(self):
        super(IntelligentGame, self).round_over()

        if self.trick_nr == 13:
            self.score()
            scores = [score/SCORE_SCALAR for score in self.player_scores]
            print("final scalar scores", scores)

            for idx, memory in enumerate(self._short_memory):
                self._short_memory[idx][-1] = scores
