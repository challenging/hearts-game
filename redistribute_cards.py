import sys
import copy
import time

import numpy as np


def get_fixed_cards(cards, must_have):
    fixed_cards = {}

    for player_idx, cardss in must_have.items():
        fixed_cards.setdefault(player_idx, [])
        for card in cardss:
            if card in cards:
                fixed_cards[player_idx].append(card)
                cards.remove(card)

    return fixed_cards


def get_hand_cards_number(position, hand_cards, trick, fixed_cards):
    numbers = {0: 0, 1: 0, 2: 0, 3: 0}

    for idx in range(1, len(trick)+1):
        player_idx = (position+(4-idx))%4
        numbers[player_idx] = len(hand_cards[position])-1-len(fixed_cards.get(player_idx, []))

    for idx in range(1, 4-len(trick)):
        player_idx = (position+idx)%4
        numbers[player_idx] = len(hand_cards[position])-len(fixed_cards.get(player_idx, []))

    del numbers[position]

    return numbers


def simple_redistribute_cards(position, hand_cards, cards, fixed_cards, numbers):
    #fixed_cards = get_fixed_cards(cards, must_have)
    #numbers = get_hand_cards_number(position, hand_cards, trick, fixed_cards)

    while True:
        copy_cards = copy.deepcopy(hand_cards)
        remaining_cards = copy.deepcopy(cards)

        np.random.shuffle(remaining_cards)

        prev_number = 0
        for player_idx, number in numbers.items():
            if player_idx != position:
                copy_cards[player_idx] = remaining_cards[prev_number:prev_number+number]
                copy_cards[player_idx].extend(fixed_cards.get(player_idx, []))

                prev_number += number

        yield copy_cards


def redistribute_cards(seed, position, hand_cards, trick, cards, must_have={}, void_info={}):
    np.random.seed(seed)

    fixed_cards = get_fixed_cards(cards, must_have)
    numbers = get_hand_cards_number(position, hand_cards, trick, fixed_cards)
    #print("---->", fixed_cards, numbers, cards)

    if np.sum([void for info in void_info.values() for void in info.values()]) == 0:
        for hand_cards in simple_redistribute_cards(position, hand_cards, cards, fixed_cards, numbers):
            yield hand_cards
    elif len(cards) < 9:
        for hand_cards in simple_redistribute_cards(position, hand_cards, cards, fixed_cards, numbers):
            yield hand_cards
    else:
        stime = time.time()

        sorted_players = []
        for player_idx in range(4):
            if player_idx != position:
                sorted_players.append([player_idx, sum(void_info[player_idx].values())])

        sorted_players = sorted(sorted_players, key=lambda x: -x[1])
        while True:
            copy_cards = copy.deepcopy(hand_cards)

            remaining_cards = copy.deepcopy(cards)
            np.random.shuffle(remaining_cards)

            for player_idx, _ in sorted_players:
                copy_cards[player_idx].extend(fixed_cards.get(player_idx, []))

                removed_cards = []
                for card in remaining_cards:
                    if void_info[player_idx][card.suit] == False:
                        copy_cards[player_idx].append(card)
                        removed_cards.append(card)

                        if len(copy_cards[player_idx]) == numbers[player_idx]+len(fixed_cards.get(player_idx, [])):
                            break

                for card in removed_cards:
                    remaining_cards.remove(card)

            is_finished = True
            while remaining_cards:
                #print("remaining_cards", remaining_cards)
                #removed_cards = []
                for card in remaining_cards:
                    #print("start to handle", card, copy_cards, fixed_cards, numbers)
                    for player_idx in [player_idx for player_idx, number in numbers.items() if (len(copy_cards[player_idx])-len(fixed_cards.get(player_idx, []))) != number]:
                        #print("find void_player", player_idx)

                        targeted_player_idx = np.random.choice(list(numbers.keys()))
                        if player_idx != targeted_player_idx and void_info[targeted_player_idx][card.suit] == False:
                            switched_card = np.random.choice(copy_cards[targeted_player_idx])
                            if switched_card not in fixed_cards.get(targeted_player_idx, []) and void_info[player_idx][switched_card.suit] == False:
                                #print("---> player-{}'s {} card vs. player-{}'s {} card".format(player_idx, card, targeted_player_idx, switched_card))
                                copy_cards[player_idx].append(switched_card)

                                copy_cards[targeted_player_idx].remove(switched_card)
                                copy_cards[targeted_player_idx].append(card)

                                removed_cards.append(card)

                                break

                        if time.time()-stime <= 0.2:
                            is_finished = False

                            break

                    if not is_finished or time.time()-stime <= 0.2:
                        is_finished = False

                        break

                if not is_finished or time.time()-stime <= 0.2:
                    is_finished = False

                    break

                for card in removed_cards:
                    if card in remaining_cards: remaining_cards.remove(card)

            if is_finished:
                yield copy_cards
