"""This module containts the abstract class Player and some implementations."""
from random import shuffle
from collections import defaultdict

from card import Suit, Rank, Card, Deck
from rules import is_card_valid


class Player(object):

    """
    Abstract class defining the interface of a Computer Player.
    """

    def __init__(self, verbose=False):
        self.seen_cards = []
        self.freeze_cards = []
        self.transfer_cards = defaultdict(list)

        self.name = None
        self.position = None

        self.verbose = verbose


    def say(self, message, *formatargs):
        if self.verbose:
            print(message.format(*formatargs))


    def set_transfer_card(self, received_player, card):
        if isinstance(card, list):
            self.transfer_cards[received_player].extend(card)
        elif isinstance(card, Card):
            self.transfer_cards[received_player].append(card)
        else:
            raise


    def set_position(self, idx):
        self.position = idx


    def pass_cards(self, hand):
        """Must return a list of three cards from the given hand."""
        return NotImplemented


    def freeze_pass_card(self, card):
        self.freeze_cards.append(card)


    def is_pass_card(self, card):
        return any([True if card == pass_card else False for pass_card in self.freeze_cards])


    def play_card(self, hand, game):
        """
        Must return a card from the given hand.
        trick is a list of cards played so far.
        trick can thus have 0, 1, 2, or 3 elements.
        are_hearts_broken is a boolean indicating whether the hearts are broken yet.
        trick_nr is an integer indicating the current trick number, starting with 0.
        """
        return NotImplemented

    def see_played_trick(self, card):
        """
        Allows the player to have a look at all four cards in the trick being played.
        """
        self.seen_cards.append(card)


    def get_leading_suit_and_rank(self, trick):
        leading_suit = trick[0].suit
        max_rank_in_leading_suit = max([card.rank for card in trick if card.suit == leading_suit])

        return leading_suit, max_rank_in_leading_suit


    def get_valid_cards(self, hand, trick, trick_nr, are_hearts_broken):
        return [card for card in hand if is_card_valid(hand, trick, card, trick_nr, are_hearts_broken)]


    def reset(self):
        self.seen_cards = []
        self.freeze_cards = []
        self.transfer_cards = defaultdict(list)


class StupidPlayer(Player):

    """
    Most simple player you can think of.
    It just plays random valid cards.
    """

    def __init__(self, verbose=False):
        super(StupidPlayer, self).__init__(verbose=verbose)

    def pass_cards(self, hand):
        cards = []
        for card in hand:
            if len(cards) == 3:
                break
            elif self.is_pass_card(card):
                continue
            else:
                cards.append(card)

        return cards

    def play_card(self, hand, game):
        # Play first card that is valid
        shuffle(hand)
        for card in hand:
            #print(game.trick_nr, game.is_heart_broken, game.trick, card, is_card_valid(hand, trick, card, game.trick_nr, game.is_heart_broken))
            if is_card_valid(hand, game.trick, card, game.trick_nr, game.is_heart_broken):
                return card

        raise AssertionError(
            'Apparently there is no valid card that can be played. This should not happen.'
        )


class SimplePlayer(Player):

    """
    This player has a notion of a card being undesirable.
    It will try to get rid of the most undesirable cards while trying not to win a trick.
    """

    def __init__(self, verbose=False):
        super(SimplePlayer, self).__init__(verbose=verbose)


    def undesirability(self, card):
        return (
            card.rank.value
            + (10 if card.suit == Suit.spades and card.rank >= Rank.queen else 0)
        )


    def pass_cards(self, hand):
        hand.sort(key=self.undesirability, reverse=True)

        cards = []
        for card in hand:
            if len(cards) == 3:
                break
            elif self.is_pass_card(card):
                continue
            else:
                cards.append(card)

        return cards


    def play_card(self, hand, game):
        # Lead with a low card
        if not game.trick:
            card = None
            if game.trick_nr == 0:
                for card in hand:
                    if card.suit == Suit.clubs and card.rank == Rank.two:
                        break
            else:
                hand.sort(key=lambda card:
                          100 if not game.is_heart_broken and card.suit == Suit.hearts else
                          card.rank.value)

                try:
                    card = hand[0]
                except IndexError:
                    print(hand)
                    raise

            return card

        hand.sort(key=self.undesirability, reverse=True)
        self.say('Hand: {}', hand)
        self.say('Trick so far: {}', game.trick)

        # Safe cards are cards which will not result in winning the trick
        leading_suit, max_rank_in_leading_suit = self.get_leading_suit_and_rank(game.trick)

        valid_cards = self.get_valid_cards(hand, game.trick, game.trick_nr, game.is_heart_broken)

        safe_cards = [card for card in valid_cards
                      if card.suit != leading_suit or card.rank <= max_rank_in_leading_suit]

        self.say('Valid cards: {}', valid_cards)
        self.say('Safe cards: {}', safe_cards)

        if len(safe_cards) > 0:
            return safe_cards[0]
        else:
            queen_of_spades = Card(Suit.spades, Rank.queen)
            # Don't try to take a trick by laying the queen of spades
            #print(999, hand, valid_cards, hand)
            if valid_cards[0] == queen_of_spades and len(valid_cards) > 1:
                return valid_cards[1]
            else:
                return valid_cards[0]

