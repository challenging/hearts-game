import copy

from collections import deque

from rules import is_card_valid, get_rating

from game import Game


class IntelligentGame(Game):
    def __init__(self, players, buffer_size=2**15, verbose=False):
        super(IntelligentGame, self).__init__(players, verbose)

        self._short_memory = []
        self._memory = deque(maxlen=buffer_size)


    def reset(self):
        super(IntelligentGame, self).reset()

        self._short_memory = []


    def get_memory(self):
        return self._memory


    def step(self, played_card=None):
        hand_cards = self._player_hands[self.current_player_idx]

        if played_card is None:
            played_card, results = self.players[self.current_player_idx].play_card(self, simulation_time_limit=0.01)

            #print("Pick {} card from {} for this trick({})".format(played_card, hand_cards, self.trick))

        if not is_card_valid(hand_cards, self.trick, played_card, self.trick_nr, self.is_heart_broken):
            raise ValueError('{} round - Player {} ({}) played an invalid card {}({}) to the trick {}.'.format(\
                self.trick_nr, self.current_player_idx, type(self.players[self.current_player_idx]).__name__, played_card, hand_cards, self.trick))

        if played_card not in self._player_hands[self.current_player_idx]:
            raise ValueError("{} round - Not found {} card in this Player-{} hand cards({})".format(\
                self.trick_nr, played_card, self.current_player_idx, self._player_hands[self.current_player_idx]))

        played_cards, probs = [], []
        for card, prob in results:
            played_cards.append(card)
            probs.append(prob)

        must_cards = [[], [], [], []]
        for player_idx, cards in self.players[self.current_player_idx].transfer_cards.items():
            must_cards[player_idx] = cards

        score_cards = copy.deepcopy(self._cards_taken)
        remaining_cards = self.players[0].get_remaining_cards(hand_cards)

        self._short_memory.append([remaining_cards[:], self.trick[:], must_cards, score_cards, played_cards, probs, self.current_player_idx])

        self._player_hands[self.current_player_idx].remove(played_card)
        self.trick.append(played_card)

        self.current_player_idx = (self.current_player_idx+1)%4
        if len(self.trick) == 4:
            self.round_over()

        for i in range(4):
            self.players[i].see_played_trick(played_card, self)


    def round_over(self):
        super(IntelligentGame, self).round_over()

        if self.trick_nr == 13:
            self.print_game_status()

            self.score()
            scores = self.player_scores

            rating = get_rating(scores)
            for idx, memory in enumerate(self._short_memory):
                self._short_memory[idx][-1] = rating

            self._memory.extend(self._short_memory)
            self._short_memory = []
