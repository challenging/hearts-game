"""This module containts the abstract class Player and some implementations."""
import os
import copy

import numpy as np

from random import shuffle, choice

from card import Suit, Rank, Card, Deck
from rules import is_card_valid


OUT_FILE = None

class Player(object):

    """
    Abstract class defining the interface of a Computer Player.
    """

    def __init__(self, verbose=False):
        self.seen_cards = []
        self.freeze_cards = []
        self.transfer_cards = {}

        self.name = None
        self.position = None
        self.proactive_mode = set()

        self.verbose = verbose


    def say(self, message, *formatargs):
        if self.verbose:
            global OUT_FILE

            message = message.format(*formatargs)
            if os.path.exists("/log"):
                if OUT_FILE is None: OUT_FILE = open("/log/game.log", "a")

                OUT_FILE.write("{}\n".format(message))
            else:
                print(message)


    def set_transfer_card(self, received_player, card):
        self.transfer_cards.setdefault(received_player, [])
        if isinstance(card, list):
            self.transfer_cards[received_player].extend(card)
        elif isinstance(card, Card):
            self.transfer_cards[received_player].append(card)
        else:
            raise


    def set_position(self, idx):
        self.position = idx


    def pass_cards(self, hand, round_idx):
        """Must return a list of three cards from the given hand."""
        return NotImplemented


    def freeze_pass_card(self, card):
        self.freeze_cards.append(card)


    def is_pass_card(self, card):
        return any([True if card == pass_card else False for pass_card in self.freeze_cards])


    def play_card(self, game):
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


    def undesirability(self, card, take_pig_card=False):
        additional_rank = 0
        if not take_pig_card and card.suit == Suit.spades:
            if card.rank == Rank.queen:
                additional_rank = 15
            elif card.rank > Rank.queen:
                additional_rank = 10
        elif card.suit == Suit.hearts:
            additional_rank = 2
        elif card == Card(Suit.clubs, Rank.ten):
            additional_rank += 2

        return card.rank.value + additional_rank


    def get_valid_cards(self, hand, game):
        cards = [card for card in hand if is_card_valid(hand, game.trick, card, game.trick_nr, game.is_heart_broken)]
        cards.sort(key=lambda x: self.undesirability(x, game.take_pig_card), reverse=(True if game.trick else False))

        return cards


    def get_remaining_cards(self, hand_cards):
        deck = Deck()

        remaining_cards = []
        for c in deck.cards:
            for pc in hand_cards + self.seen_cards:
                if c == pc:
                    break
            else:
                remaining_cards.append(c)

        return remaining_cards


    def simple_redistribute_cards(self, game, remaining_cards):
        #print("start to simple_redistribute_cards")

        fixed_cards = {}
        for player_idx in range(4):
            for card in self.transfer_cards.get(player_idx, []):
                if card in remaining_cards:
                    remaining_cards.remove(card)

                    fixed_cards.setdefault(player_idx, [])
                    fixed_cards[player_idx].append(card)

        shuffle(remaining_cards)
        for idx in range(len(game._player_hands)):
           if idx != self.position:
               cards = fixed_cards.get(idx, [])
               game._player_hands[idx] = np.random.choice(remaining_cards, len(game._player_hands[idx])-len(cards), replace=False).tolist()

               for used_card in game._player_hands[idx]:
                   remaining_cards.remove(used_card)

               if cards:
                   game._player_hands[idx].extend(cards)

        if remaining_cards:
           self.say("Error in redistributing cards, {}, {}, {}", type(self).__name__, remaining_cards, [len(v) for v in game._player_hands])
           raise

        return game


    def redistribute_cards(self, copy_game, copy_remaining_cards):
        retry = 3
        while retry >= 0:
            game = copy.deepcopy(copy_game)
            remaining_cards = copy.deepcopy(copy_remaining_cards)

            shuffle(remaining_cards)

            ori_size, fixed_cards = [], set()
            for idx in range(len(game._player_hands)):
                if idx != self.position:
                    size = len(game._player_hands[idx])
                    ori_size.append(size)

                    game._player_hands[idx] = []

                    for card in self.transfer_cards.get(idx, []):
                        if card in remaining_cards:
                            game._player_hands[idx].append(card)
                            remaining_cards.remove(card)

                            fixed_cards.add(card)

                    removed_cards = []
                    for card in remaining_cards:
                        if game.lacking_cards[idx][card.suit] == False:
                            game._player_hands[idx].append(card)
                            removed_cards.append(card)

                            if len(game._player_hands[idx]) == size:
                                break

                    for card in removed_cards:
                        remaining_cards.remove(card)
                else:
                    ori_size.append(len(game._player_hands[idx]))


            retry2 = 3
            lacking_idx = [idx for idx in range(4) if len(game._player_hands[idx]) < ori_size[idx]]
            while retry2 >= 0 and any([ori_size[player_idx] != len(game._player_hands[player_idx]) for player_idx in range(4)]):
                removed_cards = []
                for card in remaining_cards:
                    latest_lacking_idx = [idx for idx in range(4) if len(game._player_hands[idx]) < ori_size[idx]]

                    player_idx = choice([player_idx for player_idx in range(4) if player_idx not in latest_lacking_idx + [self.position]])
                    hand_cards = game._player_hands[player_idx]

                    for card_idx, hand_card in enumerate(hand_cards):
                        if hand_card not in fixed_cards and not game.lacking_cards[latest_lacking_idx[0]][hand_card.suit]:
                            game._player_hands[player_idx][card_idx] = card
                            game._player_hands[latest_lacking_idx[0]].append(hand_card)

                            removed_cards.append(card)

                            break

                for card in removed_cards:
                    remaining_cards.remove(card)

                for player_idx, size in enumerate(ori_size):
                    if len(game._player_hands[player_idx]) > size:
                        for idx in range(len(game._player_hands[player_idx])-size):
                            candidated_cards = [card for card in game._player_hands[player_idx] if card not in fixed_cards]
                            if candidated_cards:
                                card = choice(candidated_cards)
                                game._player_hands[player_idx].remove(card)
                                remaining_cards.append(card)

                retry2 -= 1

            if remaining_cards or any([ori_size[player_idx] != len(game._player_hands[player_idx]) for player_idx in range(4)]):
                retry -= 1
            else:
                copy_game = game

                break
        else:
            self.simple_redistribute_cards(copy_game, copy_remaining_cards)

        return copy_game


    def reset(self):
        self.seen_cards = []
        self.freeze_cards = []
        self.transfer_cards = {}

        self.proactive_mode = set()


class StupidPlayer(Player):

    """
    Most simple player you can think of.
    It just plays random valid cards.
    """

    def __init__(self, verbose=False):
        super(StupidPlayer, self).__init__(verbose=verbose)

    def pass_cards(self, hand, round_idx):
        cards = []
        for card in hand:
            if len(cards) == 3:
                break
            elif self.is_pass_card(card):
                continue
            else:
                cards.append(card)

        return cards

    def play_card(self, game):
        hand = game._player_hands[game.current_player_idx]

        # Play first card that is valid
        shuffle(hand)
        for card in hand:
            if is_card_valid(hand, game.trick, card, game.trick_nr, game.is_heart_broken):
                return card

        game.verbose = True
        game.print_game_status()
        raise AssertionError('Apparently there is no valid card that can be played. This should not happen.')


class SimplePlayer(Player):

    """
    This player has a notion of a card being undesirable.
    It will try to get rid of the most undesirable cards while trying not to win a trick.
    """

    def __init__(self, verbose=False):
        super(SimplePlayer, self).__init__(verbose=verbose)


    def pass_cards(self, hand, round_idx):
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


    def play_card(self, game):
        hand = game._player_hands[game.current_player_idx]

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
        #self.say('Hand: {}', hand)
        #self.say('Trick so far: {}', game.trick)

        # Safe cards are cards which will not result in winning the trick
        leading_suit, max_rank_in_leading_suit = self.get_leading_suit_and_rank(game.trick)

        valid_cards = self.get_valid_cards(hand, game)

        safe_cards = [card for card in valid_cards
                      if card.suit != leading_suit or card.rank <= max_rank_in_leading_suit]

        self.say('Valid cards: {}, Safe cards: {}', valid_cards, safe_cards)

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


class MaxCardPlayer(SimplePlayer):
    def play_card(self, game):
        hand = game._player_hands[game.current_player_idx]

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
                    raise

            return card

        leading_suit, max_rank_in_leading_suit = self.get_leading_suit_and_rank(game.trick)

        hand.sort(key=self.undesirability, reverse=True)
        valid_cards = self.get_valid_cards(hand, game)

        non_heart_cards = []
        for played_card in valid_cards:
            if played_card.suit == leading_suit and played_card.rank > max_rank_in_leading_suit:
                played_card = valid_cards[0]

                break
            else:
                if played_card.suit != Suit.hearts:
                    non_heart_cards.append(played_card)
        else:
            if non_heart_cards:
                played_card = non_heart_cards[-1]
            else:
                played_card = valid_cards[-1]

        return played_card


class MinCardPlayer(SimplePlayer):
    def play_card(self, game):
        hand = game._player_hands[game.current_player_idx]

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

        hand.sort(key=self.undesirability)

        valid_cards = self.get_valid_cards(hand, game)
        played_card = valid_cards[0]

        return played_card
