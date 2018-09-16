import sys

from rules import is_card_valid, is_score_card, card_points, reversed_score
from game import Game

from nn_utils import played_prob_to_v


class AlphaGame(Game):
    def __init__(self, players, buffer_size=2**15, verbose=False):
        super(AlphaGame, self).__init__(players, verbose)


    def step(self, played_card=None):
        player_hand = self._player_hands[self.current_player_idx]

        if played_card is None:
            played_card, played_probs = self.players[self.current_player_idx].play_card(self, return_prob=True)

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
