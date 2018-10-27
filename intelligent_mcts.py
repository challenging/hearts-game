"""This module containts the abstract class Player and some implementations."""
import copy
import time

from random import shuffle, randint, choice
from statistics import mean
from math import sqrt

from card import Suit, Rank, Card, Deck
from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, bitmask_to_card, str_to_bitmask, translate_hand_cards

from rules import get_rating

from simulated_player import TIMEOUT_SECOND

from step_game import StepGame
from redistribute_cards import redistribute_cards

from mcts import TreeNode, MCTS
from mcts import say


class IntelligentTreeNode(TreeNode):
    def __init__(self, parent, prior_p, self_player_idx, player_idx):
        super(IntelligentTreeNode, self).__init__(parent, prior_p, self_player_idx, player_idx)


    def get_value(self, c_puct):
        self._u = (c_puct * self._P * sqrt(self._parent._n_visits) / (1 + self._n_visits))

        return self._Q/(1e-16+self._n_visits) + self._u


class IntelligentMCTS(MCTS):
    def __init__(self, policy_value_fn, self_player_idx, c_puct=5):
        super(IntelligentMCTS, self).__init__(policy_value_fn, self_player_idx, c_puct)


    def _playout(self, trick_nr, state, selection_func):
        node = self.start_node
        while not state.is_finished:
            if node.is_leaf():
                break

            is_all_traverse = True

            valid_cards = state.get_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1)
            valid_moves = translate_hand_cards(valid_cards, is_bitmask=True)
            for card in valid_moves:
                if card not in node._children or node._children[card]._P == 0:
                    is_all_traverse = False

                    break

            if not is_all_traverse:
                break

            is_found = False
            for played_card, n in node.select(self._c_puct):
                suit, rank = played_card[0], played_card[1]

                if valid_cards.get(suit, 0) & rank:
                    state.step(trick_nr, selection_func, played_card)
                    is_found = True

                    node = n

                    break

            if not is_found:
                break

        # Check for end of game
        scores = [0, 0, 0, 0]
        if not state.is_finished:
            probs, scores = self._policy(trick_nr, state)
            action_probs = zip([(card.suit.value, 1 << (card.rank.value-2)) for card in Deck().cards], probs)

            node.expand(state.start_pos, action_probs)
        else:
            scores = get_rating(state.score()[0])

        node.update_recursive(scores)


    def get_move(self, 
                 hand_cards, 
                 valid_cards,
                 remaining_cards, 
                 score_cards, 
                 init_trick, 
                 void_info, 
                 must_have, 
                 selection_func, 
                 trick_nr, 
                 is_heart_broken, 
                 expose_heart_ace, 
                 is_only_played_card=False, 
                 simulation_time_limit=TIMEOUT_SECOND-0.1,
                 is_print=False):

        stime = time.time()

        simulation_cards = redistribute_cards(randint(0, 128), 
                                              self._self_player_idx, 
                                              copy.deepcopy(hand_cards), 
                                              init_trick[-1][1], 
                                              remaining_cards, 
                                              must_have, 
                                              void_info)

        valid_cards = str_to_bitmask(valid_cards)

        for simulation_card in simulation_cards:
            max_len_cards, min_min_cards = -1, 99
            for player_idx, cards in enumerate(simulation_card):
                simulation_card[player_idx] = str_to_bitmask(cards)

                if len(cards) > max_len_cards:
                    max_len_cards = len(cards)

                if len(cards) < min_min_cards:
                    min_min_cards = len(cards)

            if max_len_cards-min_min_cards < 2:
                try:
                    sm = StepGame(trick_nr,
                                  position=self._self_player_idx, 
                                  hand_cards=simulation_card,
                                  void_info=void_info,
                                  score_cards=copy.deepcopy(score_cards), 
                                  is_hearts_broken=is_heart_broken, 
                                  expose_hearts_ace=expose_heart_ace, 
                                  tricks=copy.deepcopy(init_trick),
                                  must_have=must_have)

                    if len(init_trick[-1][1]) == 4:
                        sm.post_round_end()

                    selection_func = [choice(selection_func) for _ in range(4)]
                    self._playout(trick_nr, sm, selection_func)
                except Exception as e:
                    for player_idx, cards in enumerate(simulation_card):
                        say("player-{}'s hand_cards is {}", player_idx, cards)

                    import traceback
                    traceback.print_exc()


            if time.time()-stime > simulation_time_limit:
                break

        if is_only_played_card:
            if is_print:
                for k, v in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                    if v._n_visits > 0 and valid_cards.get(k[0], 0) & k[1]:
                        say("{}: {} times, percentage: {:.4f}%", bitmask_to_card(k[0], k[1]), v._n_visits, v._P*100)

                    if v._P == 0:
                        break

            for played_card, node in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                if node._n_visits > 0 and node._P > 0 and valid_cards.get(played_card[0], 0) & played_card[1]:
                    return played_card
        else:
            results = {}
            for played_card, node in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                #if node._P > 0 and node._n_visits > 0 and valid_cards.get(played_card[0], 0) & played_card[1]:
                results.setdefault(played_card, 0)
                results[played_card] = node._n_visits

            return results


    def __str__(self):
        return "IntelligentMCTS"
