"""This module containts the abstract class Player and some implementations."""
from random import shuffle

from card import Suit, Rank, Card, Deck
from rules import is_card_valid
from player import SimplePlayer


class GreedyHeusisticPlayer(SimplePlayer):

    """
    1. If the player is the first player in the round, pick the smallest card
    2. Otherwise
        a. If the player does not need to play the same suit as the first player
            1. If the player can play Q of Spade, pick Q of Spade
            2. If the player can play hearts, pick the largest card of hearts
            3. Otherwise, pick the largest card
        b. Otherwise
            1. if the player can pick a smaller card than the first card, pick the largest that is smaller than the first card
            2. Otherwise,
                a. If the player is not the last player, pick the smallest card
                b. Otherwise, pick the largest card
    """

    def __init__(self, verbose=False):
        super(GreedyHeusisticPlayer, self).__init__(verbose)


    def undesirability(self, card):
        additional_score = 0
        if card.suit == Suit.spades:
            if card.rank == Rank.queen:
                additional_score = 50
            elif card.rank > Rank.queen:
                additional_score = 5
            elif card.suit == Suit.hearts:
                additional_score = 3

        return card.rank.value + additional_score


    def play_card(self, hand, trick, game):
        hand.sort(key=self.undesirability, reverse=True)

        if self.verbose:
            self.say('Hand: {}', hand)
            self.say('Trick so far: {}', trick)

        valid_cards = self.get_valid_cards(hand, trick, game.trick_nr, game.is_heart_broken)
        if not trick:
            #print("1.", valid_cards, valid_cards[-1])
            return valid_cards[-1]
        else:
            leading_suit, max_rank = self.get_leading_suit_and_rank(trick)

            is_same_suit = False
            for card in valid_cards:
                if card.suit == leading_suit:
                    is_same_suit = True

                    break

            if is_same_suit:
                min_card = None#Card(leading_suit, max_rank)
                hand_min_card = None
                max_card = None
                for card in valid_cards:
                   if card.rank < Card(leading_suit, max_rank).rank:
                       if min_card is None:
                           min_card = card
                       else:
                           if card.rank > min_card.rank:
                               min_card = card

                   if hand_min_card is None:
                       hand_min_card = card
                   else:
                       if hand_min_card.rank > card.rank:
                           hand_min_card = card

                   if max_card is None:
                       max_card = card
                   else:
                       if max_card.rank < card.rank:
                           max_card = card

                if min_card is not None: #!= Card(leading_suit, max_rank):
                    #print("min_card", min_card)
                    return min_card
                else:
                    #print("2. max_card", trick, valid_cards, max_card)
                    if len(trick) != 4:
                        #print("hand_min_card", hand_min_card)
                        return hand_min_card
                    else:
                        #print("max_card", max_card)
                        return max_card
            else:
                #print("2.", is_same_suit, trick, valid_cards, valid_cards[0])
                return valid_cards[0]



class DynamicRankPlayer(GreedyHeusisticPlayer):
    def __init__(self, verbose=False):
        super(DynamicRankPlayer, self).__init__(verbose)


    def undesirability(self, card):
        scores = [104, 104, 104, 104]

        for tmp_card in self.seen_cards:
            if card != tmp_card:
                scores[tmp_card.suit.value] -= tmp_card.rank.value

        additional_score = 0
        if card.suit == Suit.spades:
            if card.rank == Rank.queen:
                additional_score = 100
            elif card.rank > Rank.queen:
                additional_score = 50
            elif card.suit == Suit.hearts:
                additional_score = 40

        return card.rank.value*104/scores[card.suit.value] + additional_score
