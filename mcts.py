"""This module containts the abstract class Player and some implementations."""
import os

import copy
import time

from random import randint, choice

from card import Suit, Rank, Card, Deck, FULL_CARDS
from card import NUM_TO_INDEX, INDEX_TO_NUM, SUIT_TO_INDEX, INDEX_TO_SUIT
from card import card_to_bitmask, batch_bitmask_to_card, bitmask_to_card, str_to_bitmask, bitmask_to_str, translate_hand_cards
from card import get_remaining_cards

from rules import get_rating

from tree import TreeNode

from simulated_player import TIMEOUT_SECOND

from step_game import StepGame
from strategy_play import random_choose
from redistribute_cards import redistribute_cards


OUT_FILE = None

def say(message, *formatargs):
    global OUT_FILE

    message = message.format(*formatargs)
    if os.path.exists("/log"):
        if OUT_FILE is None: OUT_FILE = open("/log/mcts.log", "a")

        OUT_FILE.write("{}\n".format(message))
    else:
        print(message)


REMAINING_CARDS = []

def policy_value_fn(trick_nr, state):
    global REMAINING_CARDS

    if REMAINING_CARDS:
        results = REMAINING_CARDS
    else:
        results = state.get_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1, is_playout=False)

    return results, 0


class MCTS(object):
    def __init__(self, policy_value_fn, self_player_idx, c_puct=5):
        self._self_player_idx = self_player_idx

        self.start_node = TreeNode(None, 1.0, self_player_idx, None)

        self._policy = policy_value_fn
        self._c_puct = c_puct


    def _playout(self, trick_nr, state, selection_func, c_puct):
        #self.history = []

        node = self.start_node
        while not state.is_finished:
            if node.is_leaf():
                #print("is_leaf")
                break

            is_all_traverse = True

            valid_cards = state.get_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1)
            valid_moves = translate_hand_cards(valid_cards, is_bitmask=True)
            #print("current_valid_moves", valid_moves)

            for card in valid_moves:
                if card not in node._children:
                    is_all_traverse = False
                    #print("is_not_all_traverse - not")

                    break
                elif node._children[card]._P == 0:
                    is_all_traverse = False
                    #print("is_not_all_traverse - _P", card)

                    break
                #elif node._children[card]._n_visits == 0:
                #    is_all_traverse = False
                #    print("is_not_all_traverse - visits", card, valid_moves, node._children[card]._n_visits)

                #    break

            if not is_all_traverse:
                break

            current_start_pos, found_node = state.start_pos, None
            for played_card, n in node.select(c_puct):
                suit, rank = played_card[0], played_card[1]

                if valid_cards.get(suit, 0) & rank and n._P > 0:
                    state.step(trick_nr, selection_func, played_card)
                    #self.history.append((current_start_pos, bitmask_to_card(suit, rank)))

                    for sub_n in node._children.values():
                        sub_n._player_idx = current_start_pos

                    node = n

                    break
            else:
                #print("is_not_valid_card")

                break

        self._post_playout(node, trick_nr, state, selection_func)


    def _post_playout(self, node, trick_nr, state, selection_func):
        if not state.is_finished:
            action_probs, _ = self._policy(trick_nr, state)
            node.expand(state.start_pos, action_probs)

        rating = self._evaluate_rollout(trick_nr, state, selection_func)
        node.update_recursive(rating)
        #print(self.history, state.score()[0], rating)

    def _evaluate_rollout(self, trick_nr, state, selection_func):
        while not state.is_finished:
            state.step(trick_nr, selection_func)

        info = state.score()
        return get_rating(self._self_player_idx, info[0], info[1])


    def get_move(self,
                 first_player_idx,
                 hand_cards,
                 valid_cards,
                 remaining_cards,
                 score_cards,
                 num_hand_cards,
                 init_trick,
                 void_info,
                 must_have,
                 selection_func,
                 trick_nr,
                 is_heart_broken,
                 expose_heart_ace,
                 is_only_played_card=False,
                 simulation_time_limit=TIMEOUT_SECOND-0.1,
                 not_seen=False):

        stime = time.time()

        global REMAINING_CARDS
        REMAINING_CARDS = get_remaining_cards(trick_nr+len(init_trick)-1, init_trick, score_cards)
        if REMAINING_CARDS:
            self.start_node.update_recursive_percentage(REMAINING_CARDS)

        simulation_cards = redistribute_cards(randint(0, 64),
                                              self._self_player_idx,
                                              copy.deepcopy(hand_cards),
                                              num_hand_cards,
                                              init_trick[-1][1],
                                              remaining_cards,
                                              must_have,
                                              void_info,
                                              not_seen)

        vcards = str_to_bitmask(valid_cards) if not_seen else None

        c_puct = self._c_puct

        ratio, stats_shoot_the_moon = [0, 0], {}
        for simulation_card in simulation_cards:
            for player_idx, cards in enumerate(simulation_card):
                simulation_card[player_idx] = str_to_bitmask(cards)

            try:
                sm = StepGame(trick_nr,
                              position=first_player_idx,
                              hand_cards=simulation_card,
                              void_info=void_info,
                              score_cards=copy.deepcopy(score_cards),
                              is_hearts_broken=is_heart_broken,
                              expose_hearts_ace=expose_heart_ace,
                              tricks=copy.deepcopy(init_trick),
                              must_have=must_have)

                if vcards is None:
                    vcards = sm.get_valid_cards(sm.hand_cards[sm.start_pos], trick_nr+len(sm.tricks)-1)

                if len(init_trick[-1][1]) == 4:
                    sm.post_round_end()

                selection_func = [choice(selection_func) for _ in range(4)]
                self._playout(trick_nr, sm, selection_func, c_puct)

                scores, is_shoot_the_moon = sm.score()
                if is_shoot_the_moon:
                    shooter = scores.index(0)
                    stats_shoot_the_moon.setdefault(shooter, 0)
                    stats_shoot_the_moon[shooter] += 1

                ratio[0] += 1
            except Exception as e:
                ratio[1] += 1

                raise

            if time.time()-stime > simulation_time_limit:
                shooter = None
                if stats_shoot_the_moon != {}:
                    for shooter, num in stats_shoot_the_moon.items():
                        break

                if shooter:
                    say("ratio of success/failed is {}, shooter: {}, {}, {:.4f}%", \
                        ratio, shooter, num, num*100/ratio[0])
                else:
                    say("ratio of success/failed is {}", ratio)

                break

        if is_only_played_card:
            valid_cards = vcards
            vcards = [list(batch_bitmask_to_card(suit, ranks)) for suit, ranks in vcards.items()]

            if not_seen:
                for k, node in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                    if node._P > 0 and valid_cards.get(k[0], 0) & k[1]:
                        say("seen: {}, valid_cards: {}, {}-->{}: {} times, percentage: {:.4f}%, value: {:.4f}", \
                            not not_seen, vcards, node._player_idx, bitmask_to_card(k[0], k[1]), node._n_visits, node._P*100, \
                            node.get_value(self._c_puct))

                        """
                        for child_k, child_node in sorted(node._children.items(), key=lambda x: -x[1]._n_visits):
                            say("\t{}-->{}: {} times, percentage: {:.4f}%", \
                                child_node._player_idx, bitmask_to_card(child_k[0], child_k[1]), child_node._n_visits, child_node._P*100)
                        """
                    elif node._P == 0:
                        continue

            for played_card, node in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                if node._P > 0 and valid_cards.get(played_card[0], 0) & played_card[1]:
                    return played_card
        else:
            results = {}
            for played_card, node in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                results.setdefault(played_card, 0)
                results[played_card] = node._n_visits

            return results


    def update_with_move(self, last_move):
        if last_move in self.start_node._children:
            self.start_node = self.start_node._children[last_move]
        else:
            self.start_node = TreeNode(None, 1.0, self._self_player_idx, None)
            say("player-{} resets the root because not found {}", self._self_player_idx, last_move)


    def print_tree(self, node=None, card=None, depth=0):
        node = self.start_node if node is None else node

        if node._parent:
            card_str = bitmask_to_str(card[0], card[1])

            say("{} {} {}, percentage: {:.4f}, visits: {}, value: {:.4f}, {}", \
                "****"*depth, card_str, node, node._P, node._n_visits, node.get_value(self._c_puct), node._parent)

        for card, children in node._children.items():
            self.print_tree(children, card, depth+1)


    def __str__(self):
        return "MCTS"
