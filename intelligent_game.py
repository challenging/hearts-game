import sys

from nn_utils import print_a_memory, card2v, log_softmax
from rules import is_card_valid, is_score_card

from game import Game


IS_DEBUG = False


class IntelligentGame(Game):
    def __init__(self, players, buffer_size=2**15, simulation_time_limit=1, verbose=False, out_file=sys.stdout):
        super(IntelligentGame, self).__init__(players, verbose, out_file=out_file)

        self.simulation_time_limit = simulation_time_limit


    def reset(self):
        super(IntelligentGame, self).reset()

        self.trick_cards = []
        for _ in range(13):
            self.trick_cards.append([None, None, None, None])

        self.score_cards = [[], [], [], []]

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

        possible_cards = [[], [], [], []]
        possible_cards[self.current_player_idx] = hand_cards

        for player_idx, info in self.players[self.current_player_idx].void_info.items():
            if player_idx != self.current_player_idx:
                possible_cards[player_idx] = self.players[self.current_player_idx].get_remaining_cards(hand_cards)[:]
                for suit, is_void in sorted(info.items(), key=lambda x: x[0]):
                    if is_void:
                        for card in possible_cards[player_idx][:]:
                            if card.suit == suit:
                                possible_cards[player_idx].remove(card)
                                #print("try to remove {} card from player-{}, current_suit is {}, {}".format(\
                                #    card, player_idx, suit, info))

        for player_idx, cards in self.players[self.current_player_idx].transfer_cards.items():
            for idx in range(len(possible_cards)):
                if idx != player_idx:
                    for card in cards:
                        if card in possible_cards[idx]:
                            possible_cards[idx].remove(card)

        trick_cards = self.trick

        valid_cards, probs = [], [0]*52
        for card, prob in results:
            valid_cards.append(card)
            probs[card2v(card)] = prob

        probs = log_softmax(probs)

        leading_cards = len(self.trick) == 0
        expose_cards = self.expose_info

        self._short_memory.append([self.current_player_idx,
                                   self.trick_cards,
                                   self.score_cards,
                                   possible_cards,
                                   trick_cards,
                                   valid_cards,
                                   leading_cards,
                                   expose_cards,
                                   probs,
                                   None])

        if IS_DEBUG:
            print_a_memory(self._short_memory[-1])

        self._player_hands[self.current_player_idx].remove(played_card)
        self.trick.append(played_card)

        if len(self.trick) == 4:
            for idx, card in zip(range(4, 0, -1), self.trick[::-1]):
                player_idx = (self.current_player_idx+idx)%4
                self.trick_cards[self.trick_nr][player_idx] = card

        for i in range(4):
            self.players[i].see_played_trick(played_card, self)

        self.current_player_idx = (self.current_player_idx+1)%4

        if len(self.trick) == 4:
            self.round_over()


    def round_over(self):
        super(IntelligentGame, self).round_over()

        for player_idx, cards in enumerate(self._cards_taken):
            self.score_cards[player_idx] = cards

        if self.trick_nr == 13:
            self.score()

            results = [[], [], [], []]
            for player_idx, cards in enumerate(self._cards_taken):
                for card in cards:
                    if is_score_card(card):
                        results[player_idx].append(card)

            for idx, memory in enumerate(self._short_memory):
                self._short_memory[idx][-1] = results

            if IS_DEBUG:
                print_a_memory(self._short_memory[-1])
