import sys

from pprint import pprint

from card import Suit, Rank, Card, Deck
from card import SPADES_Q, SPADES_K, SPADES_A

from rules import is_card_valid, is_score_card, card_points, reversed_score
from game import Game

from nn_utils import played_prob_to_v


class AlphaGame(Game):
    def __init__(self, players, buffer_size=2**15, verbose=False):
        super(AlphaGame, self).__init__(players, verbose)

        self._memory = []


    def reset(self):
        super(AlphaGame, self).reset()

        self._memory = []


    def get_memory(self):
        return self._memory


    def score_func(self, scores, position):
        rating = [0, 0, 0, 0]

        info = zip(range(4), scores)
        pre_score, pre_rating, max_score = None, None, np.array(scores)/np.max(scores)
        for rating_idx, (player_idx, score) in enumerate(sorted(info, key=lambda x: x[1])):
            tmp_rating = rating_idx
            if pre_score is not None:
                if score == pre_score:
                    tmp_rating = pre_rating

            rating[player_idx] = ((4-tmp_rating) + (1-max_score[player_idx]))/5

            pre_score = score
            pre_rating = tmp_rating

        return rating[position]

        """
        min_score, second_score = None, None
        for idx, score in enumerate(sorted(scores)):
            if idx == 0:
                min_score = score
            elif idx == 1:
                second_score = score
                break

        self_score = scores[position]
        if self_score == min_score:
            return self_score-second_score
        else:
            return self_score-min_score
        """


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


    def step(self, played_card=None):
        player_hand = self._player_hands[self.current_player_idx]

        if played_card is None:
            played_card, played_probs = self.players[self.current_player_idx].play_card(player_hand, self, return_prob=True)

        if not is_card_valid(player_hand, self.trick, played_card, self.trick_nr, self.is_heart_broken):
            raise ValueError('Player {} ({}) played an invalid card {} to the trick {}.'.format(\
                self.current_player_idx, type(self.players[self.current_player_idx]).__name__, played_card, self.trick))

        if played_card not in self._player_hands[self.current_player_idx]:
            raise ValueError("Not found {} card in this Player-{} hand cards({})".format(\
                played_card, self.current_player_idx, self._player_hands[self.current_player_idx]))

        # store the self-play data: (state, mcts_probs, z) for training
        cards, probs = played_prob_to_v(played_probs)
        self._memory.append([self.current_status(), cards, probs, played_card, self.current_player_idx])

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

            for idx, memory in enumerate(self._memory):
                memory[-1] = self.score_func(scores, memory[-1])
