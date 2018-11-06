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

from tree import TreeNode, LevelNode

from simulated_player import TIMEOUT_SECOND

from step_game import StepGame
from redistribute_cards import redistribute_cards

from mcts import say, MCTS


class LevelMCTS(MCTS):
    def __init__(self, policy_value_fn, self_player_idx, c_puct=5):
        self._self_player_idx = self_player_idx

        self.start_node = LevelNode(None, self_player_idx, 0)
        self.last_move = None

        self._policy = policy_value_fn
        self._c_puct = c_puct


    def _playout(self, trick_nr, state, selection_func, c_puct):
        node = self.start_node
        while not state.is_finished:
            print("level", node.level)

            if node.is_leaf():
                print("is_leaf")
                break

            is_all_traverse = True

            valid_cards = state.get_valid_cards(state.hand_cards[state.start_pos], trick_nr+len(state.tricks)-1)
            valid_moves = translate_hand_cards(valid_cards, is_bitmask=True)
            #print("current_valid_moves", valid_moves)

            for card in valid_moves:
                if card not in node._probs:
                    is_all_traverse = False
                    print("is_not_all_traverse - not")

                    break
                elif node._probs[card][1] == 0:
                    is_all_traverse = False
                    print("is_not_all_traverse - _P", card, node._probs)

                    break

            if not is_all_traverse:
                break

            current_start_pos, found_node = state.start_pos, None
            for played_card, n in node.select(self.last_move, c_puct):
                suit, rank = played_card[0], played_card[1]

                print(n, n.level, "n_probs")
                if valid_cards.get(suit, 0) & rank and n._probs[played_card][1] > 0:
                    state.step(trick_nr, selection_func, played_card)

                    node._child._player_idx = current_start_pos

                    node = n
                    self.last_move = played_card

                    break
            else:
                print("is_not_valid_card")

                break

        rating = get_rating(state.score()[0])
        node.update_recursive(rating)


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
        if trick_nr >= 10:
            c_puct *= 0.6
        elif trick_nr >= 6:
            c_puct *= 0.8

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
                raise
                ratio[1] += 1

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
                for k, info in sorted(self.start_node._probs.items(), key=lambda x: -x[1][2]):
                    if info[1] > 0 and valid_cards.get(k[0], 0) & k[1]:
                        say("seen: {}, valid_cards: {}, {}-->{}: {} times, percentage: {:.4f}%", \
                            not not_seen, vcards, node._player_idx, bitmask_to_card(k[0], k[1]), info[2], info[1]*100)

                        """
                        for child_k, child_node in sorted(node._children.items(), key=lambda x: -x[1]._n_visits):
                            say("\t{}-->{}: {} times, percentage: {:.4f}%", \
                                child_node._player_idx, bitmask_to_card(child_k[0], child_k[1]), child_node._n_visits, child_node._P*100)
                        """
                    elif info[2] == 0:
                        break

            for played_card, info in sorted(self.start_node._probs.items(), key=lambda x: -x[1][2]):
                if info[1] > 0 and valid_cards.get(played_card[0], 0) & played_card[1]:
                    return played_card
        else:
            results = {}
            for played_card, node in sorted(self.start_node._children.items(), key=lambda x: -x[1]._n_visits):
                results.setdefault(played_card, 0)
                results[played_card] = node._n_visits

            return results


    def update_with_move(self, last_move):
        self.last_move = last_move

        print(self.start_node, self.start_node._child)
        self.start_node = self.start_node._child


    def __str__(self):
        return "LevelMCTS"
