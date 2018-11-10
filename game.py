import sys

from card import Suit, Rank, Card, Deck
from card import SPADES_Q, SPADES_K, SPADES_A, POINT_CARDS
from card import card_to_bitmask

from rules import is_card_valid, is_score_card, card_points, reversed_score


class Game(object):
    def __init__(self, players, verbose=False, out_file=sys.stdout):
        self.verbose = verbose
        self.out_file = out_file

        if len(players) != 4:
            raise ValueError('There must be four players')

        self.players = players

        self.reset()


    def reset(self):
        self.trick_nr = 0

        self.current_player_idx = 0
        self.trick = []

        self.player_scores = [0, 0, 0, 0]

        self.expose_heart_ace = False
        self.take_pig_card = False
        self.is_heart_broken = False
        self.is_shootmoon = False

        for i in range(4):
            self.players[i].set_position(i)
            self.players[i].reset()

        deck = Deck()

        self._player_hands = list(deck.deal())
        self._cards_taken = ([], [], [], [])
        self._b_cards_taken = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self._temp_score = [0, 0, 0, 0]

        self.lacking_cards = []
        for _ in range(4):
            self.lacking_cards.append({Suit.spades: False, Suit.hearts: False, Suit.diamonds: False, Suit.clubs: False})


    def post_round_over(self, winning_index, winning_player_index):
        # Add the information for lacking cards
        leading_suit = self.trick[winning_index].suit

        indexs = [0, 0, 0, 0]
        for shift in range(winning_index, winning_index+4):
            indexs[winning_index] = winning_player_index

            winning_index = (winning_index+1)%4
            winning_player_index = (winning_player_index+1)%4

        for i, card in zip(indexs, self.trick):
            if card.suit != leading_suit:
                self.lacking_cards[i][leading_suit] = True

        for player_idx, cards in enumerate(self._cards_taken):
            self._temp_score[player_idx] = (True if self.count_points(cards) > 0 else False)

            if self._temp_score[player_idx]:
                for idx in range(4):
                    if idx != player_idx and self.players[idx].proactive_mode:
                        self.say("set proactive_mode({}) of Player-{} to be empty", self.players[idx].proactive_mode, idx)
                        self.players[idx].proactive_mode = set()

            self._b_cards_taken[player_idx] = card_to_bitmask(cards)


    def say(self, message, *formatargs):
        if self.verbose:
            self.out_file.write(message.format(*formatargs) + "\n")


    def print_game_status(self):
        self.say("trick_nr: {:2d}, leading_position: {}, is_heart_broken: {}, expose_hearts_ace: {}", \
            self.trick_nr, self.current_player_idx, self.is_heart_broken, self.expose_heart_ace)
        self.say("="*128)
        for player_idx, (player, hand_cards, taken_cards, lacking_cards) in enumerate(zip(self.players, self._player_hands, self._cards_taken, self.lacking_cards)):
            self.say("position: {}, name:{:18s}, lacking_info: {}, hand_cards: {}, score: {:3d}, taken_cards: {}",\
                player_idx, type(player).__name__, lacking_cards, sorted(hand_cards), self.count_points(taken_cards), sorted([card for card in taken_cards if is_score_card(card)]))


    def print_hand_cards(self):
        for player_idx, (player, hand_cards, taken_cards) in enumerate(zip(self.players, self._player_hands, self._cards_taken)):
            self.say("position: {}, name:{:18s}, proactive_mode: {:6s}, hand_cards: {}, score: {:3d}, taken_cards: {}",
                player_idx,\
                type(player).__name__,\
                str(player.proactive_mode) if player.proactive_mode else "",\
                sorted(hand_cards),\
                self.count_points(taken_cards),\
                sorted([card for card in taken_cards if is_score_card(card)]))


    def are_hearts_broken(self):
        """
        Return True if the hearts are broken yet, otherwise return False.
        """
        for cards in self._cards_taken:
            if not self.is_heart_broken and any(card.suit == Suit.hearts for card in cards):
                self.is_heart_broken = True

            if not self.take_pig_card and any(card.suit == Suit.spades and card.rank == Rank.queen for card in cards):
                self.take_pig_card = True

            if self.is_heart_broken and self.take_pig_card:
                break


    def pass_cards(self, round_idx):
        pass_cards = [[], [], [], []]
        for player_idx in range(4):
            pass_cards[player_idx] = self.players[player_idx].pass_cards(self._player_hands[player_idx], round_idx)

        if round_idx%4 == 0:
            for player_idx, cards in enumerate(pass_cards):
                next_idx = (player_idx + 1) % 4

                for card in cards:
                    self._player_hands[player_idx].remove(card)
                    self.players[player_idx].set_transfer_card(next_idx, card)
                    self._player_hands[next_idx].append(card)

                    self.players[next_idx].freeze_pass_card(card)

                self.say("Player {} gives Player {} {} cards", player_idx, next_idx, cards)
        elif round_idx%4 == 1:
            for player_idx, cards in enumerate(pass_cards):
                next_idx = (player_idx + 3) % 4

                for card in cards:
                    self._player_hands[player_idx].remove(card)
                    self.players[player_idx].set_transfer_card(next_idx, card)
                    self._player_hands[next_idx].append(card)

                    self.players[next_idx].freeze_pass_card(card)

                self.say("Player {} gives Player {} {} cards", player_idx, next_idx, cards)
        elif round_idx%4 == 2:
            for player_idx, cards in enumerate(pass_cards):
                next_idx = (player_idx + 2) % 4

                for card in cards:
                    self._player_hands[player_idx].remove(card)
                    self.players[player_idx].set_transfer_card(next_idx, card)
                    self._player_hands[next_idx].append(card)

                    self.players[next_idx].freeze_pass_card(card)

                self.say("Player {} gives Player {} {} cards",\
                        player_idx, next_idx, cards)
        else:
            pass

        self.print_hand_cards()


    def play(self, num_of_rounds=13):
        leading_index = self.player_index_with_two_of_clubs()
        self.current_player_idx = leading_index

        for _ in range(num_of_rounds):
            self.play_trick()


    def score(self):
        self.check_shootmoon()

        if self.verbose:
            self.say('Results of this game:')

        if self.is_shootmoon:
            max_score = max([self.count_points(self._cards_taken[idx]) for idx in range(4)])

            for i in range(4):
                s = self.count_points(self._cards_taken[i])

                if s > 0:
                    self.player_scores[i] = 0
                else:
                    self.player_scores[i] = max_score*4
        else:
            for i in range(4):
                self.player_scores[i] = self.count_points(self._cards_taken[i])

        for i in range(4):
            self.say('self.is_shootmoon={}, Player {} got {} points from the cards {}',
                self.is_shootmoon, i, self.player_scores[i], sorted([card for card in self._cards_taken[i] if card in POINT_CARDS]))


    def check_shootmoon(self):
        for i in range(4):
            self.is_shootmoon = reversed_score(self._cards_taken[i])
            if self.is_shootmoon:
                break


    def play_trick(self):
        """
        Simulate a single trick.
        leading_index contains the index of the player that must begin.
        """
        if not self.is_heart_broken:
            self.are_hearts_broken()

        for _ in range(4):
            self.step()

            if self.verbose:
                if len(self.trick) == 1:
                    tmp_player_idx = self.current_player_idx-1 if self.current_player_idx > 0 else 3

                    self.say("Player {}({}) played {} card as the leading card",\
                        tmp_player_idx, type(self.players[tmp_player_idx]).__name__, self.trick[0])


    def round_over(self):
        winning_index, winning_card = self.winning_index()
        winning_player_index = (self.current_player_idx + winning_index) % 4

        self._cards_taken[winning_player_index].extend(self.trick)
        self.post_round_over(winning_index, winning_player_index)

        self.trick_nr += 1

        if self.verbose:
            self.say("")
            if any([l for player_idx in range(4) for l in self.lacking_cards[player_idx].values()]):
                self.say("the information about lacking cards are")
                info = []
                for player_idx in range(4):
                    is_lacking = any([l for l in self.lacking_cards[player_idx].values()])
                    if is_lacking:
                        info.append("Player-{} lacks of {}".format(player_idx, [suit for suit, is_lacking in self.lacking_cards[player_idx].items() if is_lacking]))
                self.say("{}", ",".join(info))
                self.say("")

            self.say("the winning_player_index is {}({}, {}), is_heart_broken: {}, expose_heart_ace: {}", \
                winning_player_index, self.current_player_idx, winning_index, self.is_heart_broken, self.expose_heart_ace)
            self.say("player {}({}) win this {:2d} trick by {} card based on {}",\
                winning_player_index, type(self.players[winning_player_index]).__name__, self.trick_nr, winning_card, self.trick)
            self.say("after {:3d} round, status of every players' hand cards", self.trick_nr)
            self.say("==================================================================")
            self.print_hand_cards()

        self.current_player_idx = winning_player_index
        self.trick = []

        return winning_card


    def step(self, played_card=None):
        player_hand = self._player_hands[self.current_player_idx]

        if played_card is None:
            played_card = self.players[self.current_player_idx].play_card(self)

        if not is_card_valid(player_hand, self.trick, played_card, self.trick_nr, self.is_heart_broken):
            raise ValueError('Player {} ({}) played an invalid card {} to the trick {}.'.format(\
                self.current_player_idx, type(self.players[self.current_player_idx]).__name__, played_card, self.trick))

        if played_card not in self._player_hands[self.current_player_idx]:
            raise ValueError("Not found {} card in this Player-{} hand cards({})".format(\
                played_card, self.current_player_idx, self._player_hands[self.current_player_idx]))

        self._player_hands[self.current_player_idx].remove(played_card)
        self.trick.append(played_card)

        for i in range(4):
            self.players[i].see_played_trick(played_card, self)

        self.current_player_idx = (self.current_player_idx+1)%4
        if len(self.trick) == 4:
            self.round_over()


    def player_index_with_two_of_clubs(self):
        two_of_clubs = Card(Suit.clubs, Rank.two)
        for i in range(4):
            if two_of_clubs in self._player_hands[i]:
                return i

        for player_idx in range(4):
            self.say("Player - {}'s init_cards: {}", player_idx, self._player_hands[player_idx])

        raise AssertionError('No one has the two of clubs. This should not happen.')


    def winning_index(self):
        """
        Determine the index of the card that wins the trick.
        trick is a list of four Cards, i.e. an entire trick.
        """
        leading_suit, leading_rank = self.trick[0].suit, self.trick[0].rank

        winning_index = 0
        winning_card = self.trick[0]
        for i, card in enumerate(self.trick):
            if card.suit == leading_suit and card.rank > leading_rank:
                winning_index = i
                winning_card = card

                leading_rank = winning_card.rank

        return winning_index, winning_card


    def count_points(self, cards):
        """
        Count the number of points in cards, where cards is a list of Cards.
        """

        point, is_double = 0, 1
        for card in cards:
            point += card_points(card)*(2 if self.expose_heart_ace and card.suit == Suit.hearts else 1)

            if card == Card(Suit.clubs, Rank.ten):
                is_double = 2

        return point*is_double


    def get_game_winners(self):
        is_game_over = all([len(cards) == 0 for cards in self._player_hands])

        if is_game_over:
            self.score()
            scores = self.player_scores

            min_score = min(scores)

            return [idx for idx in range(4) if scores[idx] == min_score]
        else:
            return []


    def score_func(self, scores, position):
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
