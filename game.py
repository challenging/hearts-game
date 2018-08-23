import sys

from card import Suit, Rank, Card, Deck
from rules import is_card_valid, card_points, reversed_score


class Game:

    def __init__(self, players, verbose=False):
        """
        players is a list of four players
        """
        self.verbose = verbose
        if len(players) != 4:
            raise ValueError('There must be four players.')

        self.players = players
        self.player_scores = [0, 0, 0, 0]

        self.reset()


    def reset(self):
        self.trick_nr = 0

        self.current_player_idx = 0
        self.trick = []

        self.is_heart_broken = False
        self.is_shootmoon = False

        for i in range(4):
            self.players[i].set_position(i)
            self.players[i].reset()

        deck = Deck()

        self._player_hands = list(deck.deal())
        self._cards_taken = ([], [], [], [])

        self.lacking_cards = []
        for _ in range(4):
            suits = {}
            for suit in [Suit.spades, Suit.hearts, Suit.diamonds, Suit.clubs]:
                suits[suit] = False

            self.lacking_cards.append(suits)

        #if self.verbose:
        #    self.print_hand_cards()


    def add_lacking_cards_info(self, winning_index, winning_player_index):
        leading_suit = self.trick[winning_index].suit

        indexs = [0, 0, 0, 0]
        for shift in range(winning_index, winning_index+4):
            indexs[winning_index] = winning_player_index

            winning_index = (winning_index+1)%4
            winning_player_index = (winning_player_index+1)%4

        for i, card in zip(indexs, self.trick):
            if card.suit != leading_suit:
                self.lacking_cards[i][leading_suit] = True


    def say(self, message, *formatargs):
        if self.verbose:
            print(message.format(*formatargs))


    def print_hand_cards(self):
        if self.verbose:
            for player_idx, (player, hand_cards) in enumerate(zip(self.players, self._player_hands)):
                print(player_idx, type(player).__name__, hand_cards, self.count_points(self._cards_taken[player_idx]))
            print()


    def are_hearts_broken(self):
        """
        Return True if the hearts are broken yet, otherwise return False.
        """
        for cards in self._cards_taken:
            if any(card.suit == Suit.hearts for card in cards):
                self.is_heart_broken = True

                break


    def pass_cards(self):
        # Perform the card passing.
        # Currently always passes in one direction.
        # Alternating directions can be implemented later if desirable

        if self.verbose:
            self.print_hand_cards()

        for i in range(4):
            for card in self.players[i].pass_cards(self._player_hands[i]):
                next_idx = (i + 1) % 4

                self._player_hands[i].remove(card)
                self._player_hands[next_idx].append(card)

                self.players[next_idx].freeze_pass_card(card)

                if self.verbose:
                    print("Player {}({}) give Player {}({}) {} card".format(\
                        i, type(self.players[i]).__name__, \
                        next_idx, type(self.players[next_idx]).__name__, card))

        if self.verbose:
            print()
            self.print_hand_cards()


    def play(self, num_of_rounds=13):
        # Play the tricks
        leading_index = self.player_index_with_two_of_clubs()
        self.current_player_idx = leading_index

        for _ in range(num_of_rounds):
            self.play_trick()


    def score(self):
        self.check_shootmoon()

        if self.verbose:
            self.say('Results of this game:')

        for i in range(4):
            s = self.count_points(self._cards_taken[i])

            if self.is_shootmoon:
                if s == 26:
                    self.player_scores[i] = 0
                else:
                    self.player_scores[i] = 26
            else:
                self.player_scores[i] = s

            if self.verbose:
                self.say('self.is_shootmoon={}, Player {} got {} points from the cards {}',
                         self.is_shootmoon, i, self.player_scores[i], ' '.join(str(card) for card in self._cards_taken[i]))


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
        self.are_hearts_broken()
        for _ in range(4):
            self.step()

            if self.verbose:
                if len(self.trick) == 1:
                    tmp_player_idx = self.current_player_idx-1 if self.current_player_idx > 0 else 3

                    print("Player {}({}) played {} card as the leading card".format(\
                        tmp_player_idx, type(self.players[tmp_player_idx]).__name__, self.trick[0]))


    def round_over(self):
        winning_index, winning_card = self.winning_index()
        winning_player_index = (self.current_player_idx + winning_index) % 4

        self._cards_taken[winning_player_index].extend(self.trick)

        self.add_lacking_cards_info(winning_index, winning_player_index)

        self.trick_nr += 1

        if self.verbose:
            print()
            print("the information about lacking cards are")
            print(self.lacking_cards)
            print()
            print("the winning_player_index is {}({}, {})".format(winning_player_index, self.current_player_idx, winning_index))
            print("player {}({}) win this {:2d} trick by {} card based on {}".format(\
                winning_player_index, type(self.players[winning_player_index]).__name__, self.trick_nr, winning_card, self.trick))
            print("after {:3d} round, status of every players' hand cards".format(self.trick_nr))
            print("==================================================================")
            self.print_hand_cards()

        self.current_player_idx = winning_player_index
        self.trick = []

        return winning_card


    def step(self, played_card=None):
        player_hand = self._player_hands[self.current_player_idx]

        if played_card is None:
            played_card = self.players[self.current_player_idx].play_card(player_hand, self)

        if not is_card_valid(player_hand, self.trick, played_card, self.trick_nr, self.is_heart_broken):
            raise ValueError('Player {} ({}) played an invalid card {} to the trick {}.'.format(\
                self.current_player_idx, type(self.players[self.current_player_idx]).__name__, played_card, self.trick))

        if played_card not in self._player_hands[self.current_player_idx]:
            raise ValueError("Not found {} card in this Player-{} hand cards({})".format(\
                played_card, sele.current_player_idx, self._player_hands[self.current_player_idx]))

        self._player_hands[self.current_player_idx].remove(played_card)
        self.trick.append(played_card)

        for i in range(4):
            self.players[i].see_played_trick(self.trick[-1])

        self.current_player_idx = (self.current_player_idx+1)%4
        if len(self.trick) == 4:
            self.round_over()


    def player_index_with_two_of_clubs(self):
        two_of_clubs = Card(Suit.clubs, Rank.two)
        for i in range(4):
            if two_of_clubs in self._player_hands[i]:
                return i

        for player_idx in range(4):
            print(player_idx, self._player_hands[player_idx])

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

        return sum(card_points(card) for card in cards)


    def get_game_winners(self):
        is_game_over = all([len(cards) == 0 for cards in self._player_hands])

        if is_game_over:
            self.score()
            scores = self.player_scores

            min_score = sys.maxsize
            for score in scores:
                if score < min_score:
                    min_score = score

            return [idx for idx in range(4) if scores[idx] == min_score]
        else:
            return []
