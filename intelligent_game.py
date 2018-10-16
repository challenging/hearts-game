import sys
import time

import numpy as np

from collections import deque

from rules import is_card_valid, is_score_card, card_points, reversed_score
from game import Game


class IntelligentGame(Game):
    def __init__(self, players, buffer_size=2**15, verbose=False):
        super(AlphaGame, self).__init__(players, verbose)

        self._short_memory = []
        self._memory = deque(maxlen=buffer_size)


    def reset(self):
        super(IntelligentGame, self).reset()

        self._short_memory = []


    def get_memory(self):
        return self._memory


    def current_status(self):
        status = []

        def contains_spades(cards):
            has_queen, has_king, has_ace = False, False, False
            count = 0

            for card in cards:
                if card == SPADES_Q:
                    has_queen = True
                elif card == SPADES_K:
                    has_king = True
                elif card == SPADES_A:
                    has_ace = True
                elif card.suit == Suit.spades:
                    count += 1

            return [has_queen, has_king, has_ace, count==0, count==1, count==2, count==3, count==4, count>5]

        status.extend(contains_spades(self._player_hands[self.current_player_idx]))

        return status


    def score_func(self, scores, position):
        rating = [0, 0, 0, 0]

        info = zip(range(4), scores)
        pre_score, pre_rating, sum_score = None, None, np.array(scores)/np.sum(scores)
        for rating_idx, (player_idx, score) in enumerate(sorted(info, key=lambda x: -x[1])):
            tmp_rating = rating_idx
            if pre_score is not None:
                if score == pre_score:
                    tmp_rating = pre_rating

            rating[player_idx] = (tmp_rating/4 + (1-sum_score[player_idx]))/2

            pre_score = score
            pre_rating = tmp_rating

        return rating[position]


    def step(self, played_card=None):
        player_hand = self._player_hands[self.current_player_idx]

        if played_card is None:
            played_card, results = self.players[self.current_player_idx].play_card(self, return_prob=True)
            print("Pick {} card from {} for this trick, {}".format(played_card, player_hand, self.trick))

        if not is_card_valid(player_hand, self.trick, played_card, self.trick_nr, self.is_heart_broken):
            raise ValueError('{} round - Player {} ({}) played an invalid card {}({}) to the trick {}.'.format(\
                self.trick_nr, self.current_player_idx, type(self.players[self.current_player_idx]).__name__, played_card, player_hand, self.trick))

        if played_card not in self._player_hands[self.current_player_idx]:
            raise ValueError("{} round - Not found {} card in this Player-{} hand cards({})".format(\
                self.trick_nr, played_card, self.current_player_idx, self._player_hands[self.current_player_idx]))

        cards, probs, values = [], [], []
        for card, info in results.items():
            cards.append(card)

            probs.append(info[0])
            values.append(info[1])

        self._short_memory.append([cards, probs, values, self.current_player_idx])

        self._player_hands[self.current_player_idx].remove(played_card)
        self.trick.append(played_card)

        for i in range(4):
            self.players[i].see_played_trick(self.trick[-1])

        self.current_player_idx = (self.current_player_idx+1)%4
        if len(self.trick) == 4:
            self.round_over()

            if self.trick_nr == 13:
                self.score()
                scores = self.player_scores

                for memory in enumerate(self._short_memory):
                    memory[-1] = self.score_func(scores, memory[-1])

                    self._memory.append(memory)
