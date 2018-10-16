"""This module containts the abstract class Player and some implementations."""
import sys

import copy
import time

import numpy as np
import multiprocessing as mp

from functools import cmp_to_key
from scipy.stats import describe
from collections import defaultdict
from random import shuffle, choice

from card import Suit, Rank, Card, Deck, POINT_CARDS
from player import StupidPlayer, SimplePlayer, MinCardPlayer, MaxCardPlayer


TIMEOUT_SECOND = 0.935
COUNT_CPU = 4#mp.cpu_count()


class MonteCarloPlayer(SimplePlayer):
    def __init__(self, num_of_cpu=COUNT_CPU, verbose=False):
        super(MonteCarloPlayer, self).__init__(verbose=verbose)

        self.num_of_cpu = num_of_cpu


    def winning_score_func(self, scores):
        return np.mean(scores)


    def no_choice(self, played_card):
        time.sleep(np.random.randint(9000, 9200)*0.0001)

        return played_card


    def select_card(self, game, valid_cards, winning_score):
        played_card = None

        min_score = sys.maxsize
        for card in valid_cards:
            score = self.winning_score_func(winning_score[card])

            if score < min_score:
                min_score = score
                played_card = card

            stats = describe(winning_score[card])
            self.say("{} {}, simulation: {} round --> valid_cards: {}, simulate {} card --> average_score {:.3f}, (mean={:.2f}, std={:.2}, minmax={})",
                game.trick_nr,
                type(self).__name__,
                len(winning_score[card]),
                valid_cards,
                card,
                np.mean(winning_score[card]),
                stats.mean,
                stats.variance**0.5,
                stats.minmax)

        return played_card


    def play_card(self, game, other_info={}, simulation_time_limit=TIMEOUT_SECOND):
        btime = time.time()

        for player_idx, suits in other_info.get("lacking_info", {}).items():
            for suit in suits:
                game.lacking_cards[player_idx][suit] = True

            self.say("Player-{} may lack of {} suit({}, {})", player_idx, suit, game.lacking_cards[player_idx], other_info)

        self.say("Player-{}, the information of lacking_card is {}", self.position, [(player_idx, k) for player_idx, info in enumerate(game.lacking_cards) for k, v in info.items() if v])

        game.are_hearts_broken()

        hand_cards = game._player_hands[self.position]
        valid_cards = self.get_valid_cards(hand_cards, game)

        card = None
        if len(valid_cards) > 1:
            stime = time.time()

            winning_score = defaultdict(list)
            pool = mp.Pool(processes=self.num_of_cpu)
            while time.time() - stime < simulation_time_limit:
                mul_result = [pool.apply_async(self.run_simulation, args=(game, hand_cards, card)) for card in valid_cards]
                results = [res.get() for res in mul_result]

                for card, score in results:
                    if score is not None:
                        winning_score[card].append(score)

            pool.close()

            played_card = self.select_card(game, valid_cards, winning_score)
        else:
            played_card = self.no_choice(valid_cards[0])

        self.say("pick {} card, cost {:.8} seconds", played_card, time.time()-btime)

        return played_card


    def score_func(self, scores):
        return scores[self.position]


    def get_players(self, game):
        return [SimplePlayer() for _ in range(4)]


    def overwrite_game_rule(self, current_trick_nr, game):
        pass


    def evaluate_proactive_mode(self, hands):
        pass


    def run_simulation(self, game, hand_cards, played_card):
        self_score = None

        try:
            remaining_cards = self.get_remaining_cards(hand_cards)

            current_trick_nr = game.trick_nr
            game.verbose = False
            game.players = self.get_players(game)

            game = self.redistribute_cards(game, remaining_cards[:])

            game.step(played_card)

            for _ in range(4-len(game.trick)):
                game.step()

            for _ in range(13-game.trick_nr):
                game.play_trick()

            self.overwrite_game_rule(current_trick_nr, game)

            game.score()

            self_score = self.score_func(game.player_scores)
        except:
            pass

        return played_card, self_score


class MonteCarloPlayer3(MonteCarloPlayer):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer3, self).__init__(verbose=verbose)


    def score_func(self, scores):
        min_score, other_score = None, 0
        for idx, score in enumerate(sorted(scores)):
            if idx == 0:
                min_score = score
            else:
                other_score += score

        self_score = scores[self.position]
        if self_score == min_score:
            return self_score-other_score/3
        else:
            return self_score-min_score

        """
        rating = [0, 0, 0, 0]

        info = list(zip(range(4), scores))
        pre_score, pre_rating, norm_score = None, None, np.array(scores)/np.sum(scores)
        for rating_idx, (player_idx, score) in enumerate(sorted(info, key=lambda x: -x[1])):
            tmp_rating = rating_idx
            if pre_score is not None:
                if score == pre_score:
                    tmp_rating = pre_rating

            rating[player_idx] = (4-tmp_rating)/4 + norm_score[player_idx]

            pre_score = score
            pre_rating = tmp_rating

        return rating[self.position]
        """


class MonteCarloPlayer4(MonteCarloPlayer3):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer4, self).__init__(verbose=verbose)


    def overwrite_game_rule(self, current_trick_nr, game):
        if current_trick_nr >= 6:
            return

        import types

        def score(self):
            self.check_shootmoon()

            if self.is_shootmoon:
                max_score = max([self.count_points(self._cards_taken[idx]) for idx in range(4)])

                for i in range(4):
                    s = self.count_points(self._cards_taken[i])

                    if s > 0:
                        self.player_scores[i] = 0
                    else:
                        self.player_scores[i] = max_score
            else:
                for i in range(4):
                    self.player_scores[i] = self.count_points(self._cards_taken[i])
                    if self.player_scores[i] == 0:
                        for card in self._cards_taken[i]:
                            if card == Card(Suit.clubs, Rank.ten):
                                self.player_scores[i] = 2

                                break

        game.score = types.MethodType(score, game)


class MonteCarloPlayer5(MonteCarloPlayer4):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer5, self).__init__(verbose=verbose)


    def set_proactive_mode(self, hand, round_idx):
        hand.sort(key=lambda x: self.undesirability(x), reverse=True)

        hand_cards = {Suit.spades: [], Suit.hearts: [], Suit.diamonds: [], Suit.clubs: []}
        for card in hand:
            hand_cards[card.suit].append(max(card.rank.value-10, 0))

        pass_low_card = False

        point_of_suit = 0
        for suit, cards in hand_cards.items():
            point_of_suit = np.sum(cards)
            if suit == Suit.hearts:
                if (point_of_suit > 7 and len(cards) > 5):# or (point_of_suit > 5 and len(cards) > 4) or (point_of_suit > 4 and len(cards) > 5):
                    self.proactive_mode.add(suit)
            else:
                if (point_of_suit > 6 and len(cards) > 4) and (len(hand_cards[Suit.hearts]) > 2 and np.sum(hand_cards[Suit.hearts]) > 3):
                    self.proactive_mode.add(suit)
                elif (point_of_suit > 5 and len(cards) > 5) and (len(hand_cards[Suit.hearts]) > 2 and np.sum(hand_cards[Suit.hearts]) > 3):
                    self.proactive_mode.add(suit)
                elif (point_of_suit > 4 and len(cards) > 6) and (len(hand_cards[Suit.hearts]) > 2 and np.sum(hand_cards[Suit.hearts]) > 3):
                    self.proactive_mode.add(suit)

        if not self.proactive_mode:
            points = np.sum([v for vv in hand_cards.values() for v in vv])

            if points > 18 and np.sum(hand_cards[Suit.hearts]) > 3:
                pass_low_card = True

        return hand_cards, pass_low_card


    def pass_cards(self, hand, round_idx):
        hand_cards, pass_low_card = self.set_proactive_mode(hand, round_idx)

        pass_cards, keeping_cards = [], []
        if self.proactive_mode:
            for card in hand:
                if card.suit not in self.proactive_mode:
                    if card.suit != Suit.hearts:
                        pass_cards.append(card)
                    else:
                        if card.rank < Rank.jack:
                            pass_cards.append(card)

            pass_cards.sort(key=lambda x: self.undesirability(x), reverse=False)

            if len(pass_cards) < 3:
                hand.sort(key=lambda x: self.undesirability(x), reverse=False)

                pass_cards.extend(hand[:3-len(pass_cards)])

            self.say("{} ----> proactive_mode: {}, pass cards are {}, hand_cards are {}",\
                 type(self).__name__, self.proactive_mode, pass_cards[:3], hand)
        elif pass_low_card:
            hand.sort(key=lambda x: self.undesirability(x), reverse=False)
            pass_cards = hand

            self.say("{} ----> pass low cards are {} from {}({})", type(self).__name__, pass_cards[:3], hand, hand_cards)
        else:
            pass_cards = []

            if len(hand_cards[Suit.hearts]) > 2 and len(hand_cards[Suit.hearts]) < 5:
                keeping_cards.append(Card(Suit.hearts, Rank.ace))

            num_spades = 0
            for card in hand:
                if card.suit == Suit.spades:
                    if card.rank < Rank.queen:
                        num_spades += 1

            for card in hand:
                if num_spades < 3:
                    if card.suit == Suit.spades and card.rank > Rank.jack:
                        pass_cards.append(card)
                else:
                    if card not in keeping_cards and (card.suit != Suit.spades or (card.suit == Suit.spades and card.rank > Rank.queen)):
                        pass_cards.append(card)

            if len(pass_cards) < 3:
                for card in hand:
                    if card not in keeping_cards and card not in pass_cards:
                        pass_cards.append(card)

        self.say("proactive mode: {}, keeping_cards are {}, pass card is {}", self.proactive_mode, keeping_cards, pass_cards[:3])

        return pass_cards[:3]


    def get_players(self, game):
        if Suit.hearts in self.proactive_mode:
            return [StupidPlayer() for _ in range(4)]
        else:
            return [SimplePlayer() for _ in range(4)]



def sorted_suits(xs, ys):
    if len(xs[1]) == len(ys[1]):
        if ys[0] == Suit.spades:
            return 5 - xs[0].value
        else:
           return ys[0].value - xs[0].value
    else:
        return len(xs[1]) - len(ys[1])

class MonteCarloPlayer6(MonteCarloPlayer5):
    def __init__(self, verbose=False):
        super(MonteCarloPlayer6, self).__init__(verbose=verbose)


    def pass_cards(self, hand, round_idx):
        ps = []

        num_of_suits = defaultdict(list)
        for card in hand:
            num_of_suits[card.suit].append(card)

        for card in num_of_suits.get(Suit.spades, []):
            if card.rank >= Rank.queen:
                ps.append(card)


        for suit, cards in sorted(num_of_suits.items(), key=cmp_to_key(sorted_suits)):
            if len(ps) >= 3:
                break

            if suit == Suit.spades:
                continue

            if cards[0].suit in [Suit.clubs, Suit.diamonds]:
                for card in sorted(cards, key=lambda x: -x.rank.value):
                    ps.append(card)
            elif cards[0].suit == Suit.hearts:
                for card in sorted(cards, key=lambda x: -x.rank.value):
                    if card.rank >= Rank.jack:
                        ps.append(card)

        return ps[:3]

    """
    def evaluate_proactive_mode(self, hands):
        self.say("re-evaluate the hands is suitable for the mode of shooting_the_moon, self.proactive_mode={}", self.proactive_mode)
        super(MonteCarloPlayer6, self).set_proactive_mode(hands, 3)


    def select_card(self, game, valid_cards, winning_score):
        valid_cards = sorted(valid_cards)

        if Suit.hearts in self.proactive_mode:
            if all([card.suit == Suit.hearts for card in valid_cards]):
                deck = Deck()

                remaining_cards = {}
                for card in deck.cards:
                    if card.suit == Suit.hearts and card not in self.seen_cards and card not in game.trick and card not in game._player_hands[self.position]:
                        remaining_cards.setdefault(card.suit, card)
                        if card.rank > remaining_cards[card.suit].rank:
                            remaining_cards[card.suit] = card

                for card in valid_cards[::-1]:
                    if card.suit in remaining_cards:
                        if card > remaining_cards[card.suit]:
                            self.say("1. force to get this card - {} from {} because of {}", card, valid_cards, game.trick)

                            return card


        if game.trick and Suit.hearts in self.proactive_mode:
            is_point_card_in_trick, leading_suit, current_max_rank = False, game.trick[0].suit, None
            for card in game.trick:
                if current_max_rank is None:
                    current_max_rank = card.rank
                else:
                    if card.suit == leading_suit and card.rank > current_max_rank:
                        current_max_rank = card.rank

                    if card.suit == Suit.hearts:
                        is_point_card_in_trick = True

            if is_point_card_in_trick and valid_cards[-1].suit == leading_suit and valid_cards[-1].rank > current_max_rank:
                self.say("2. force to get this card - {} from {} because of {}", valid_cards[-1], valid_cards, game.trick)

                return valid_cards[-1]


        return super(MonteCarloPlayer6, self).select_card(game, valid_cards, winning_score)
    """
