import sys

import copy
import time
import random

from random import shuffle, choice, randint

TIME_LIMIT = 0.2


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


def simple_redistribute_cards(position, hand_cards, cards, fixed_cards, numbers, random_void=False):
    num = [0, 0, 0 ,0 , 0]
    while True:
        copy_cards = copy.deepcopy(hand_cards)
        remaining_cards = copy.copy(cards)

        shuffle(remaining_cards)

        prev_number = 0
        for player_idx, number in numbers.items():
            if player_idx != position:
                copy_cards[player_idx] = remaining_cards[prev_number:prev_number+number]
                copy_cards[player_idx].extend(fixed_cards.get(player_idx, []))

                prev_number += number

        random_void_suit = 4
        if random_void:
            random_void_suit = randint(0, 4)

        if random_void_suit < 4:
            selected_player = choice([player_idx for player_idx in range(4) if player_idx != position])
            another_selected_player = choice([player_idx for player_idx in range(4) if player_idx != position and player_idx != selected_player])

            #print("player-", selected_player, copy_cards[selected_player])
            #print("player-", another_selected_player, copy_cards[another_selected_player])

            card_idx = 0
            for idx, card_1 in enumerate(copy_cards[selected_player]):
                if card_1.suit.value == random_void_suit:
                    for iidx in range(card_idx, len(copy_cards[another_selected_player])):
                        card_2 = copy_cards[another_selected_player][iidx]
                        if card_2.suit.value != random_void_suit:
                            copy_cards[selected_player][idx], copy_cards[another_selected_player][iidx] = \
                                copy_cards[another_selected_player][iidx], copy_cards[selected_player][idx]

                            #print("switch", copy_cards[selected_player][idx], copy_cards[another_selected_player][iidx])

                            break

            #print("player-", selected_player, copy_cards[selected_player])
            #print("player-", another_selected_player, copy_cards[another_selected_player])

        """
        if random_void:
            num[random_void_suit] += 1
            print(random_void_suit, num)
            print(copy_cards[selected_player])
            print(copy_cards[another_selected_player])
            print()
        """

        yield copy_cards


def redistribute_cards(seed, position, hand_cards, trick, cards, must_have={}, void_info={}):
    random.seed(seed)

    fixed_cards = get_fixed_cards(cards, must_have)
    numbers = get_hand_cards_number(position, hand_cards, trick, fixed_cards)

    random_void = False
    if len(cards) > 19:
        random_void = True

    if sum([void for info in void_info.values() for void in info.values()]) == 0:
        for hand_cards in simple_redistribute_cards(position, hand_cards, cards, fixed_cards, numbers, random_void):
            yield hand_cards
    elif len(cards) < 9:
        for hand_cards in simple_redistribute_cards(position, hand_cards, cards, fixed_cards, numbers):
            yield hand_cards
    else:
        sorted_players = []
        for player_idx in range(4):
            if player_idx != position:
                sorted_players.append([player_idx, sum(void_info[player_idx].values())])

        sorted_players = sorted(sorted_players, key=lambda x: -x[1])
        while True:
            stime = time.time()

            copy_cards = copy.deepcopy(hand_cards)

            remaining_cards = copy.deepcopy(cards)
            shuffle(remaining_cards)

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

            for card in remaining_cards[:]:
                for player_idx in range(4):
                    if card in copy_cards[player_idx]:
                        remaining_cards.remove(card)

            while remaining_cards:
                #print("remaining_cards", remaining_cards)
                for card in remaining_cards[:]:
                    #print("start to handle", card, copy_cards, fixed_cards, numbers)
                    void_players = []
                    for player_idx, number in numbers.items():
                        num_hand_cards = len(copy_cards[player_idx])-len(fixed_cards.get(player_idx, []))

                        if num_hand_cards < number:
                            void_players.append(player_idx)

                            #print("find void_player", player_idx, card, remaining_cards, len(copy_cards[player_idx]), len(fixed_cards.get(player_idx, [])), num_hand_cards, number)
                        elif num_hand_cards > number:
                            print("find void_player", player_idx, card, remaining_cards, len(copy_cards[player_idx]), len(fixed_cards.get(player_idx, [])), num_hand_cards, number)
                            raise

                    if not void_players:
                        remaining_cards.remove(card)

                        continue

                    #void_players = [player_idx for player_idx, number in numbers.items() if (len(copy_cards[player_idx])-len(fixed_cards.get(player_idx, []))) != number]
                    shuffle(void_players)
                    for player_idx in void_players:
                        #print("find void_player", player_idx, card, remaining_cards)

                        target_players = list(numbers.keys())
                        shuffle(target_players)

                        is_switched = False
                        for targeted_player_idx in target_players:
                            if player_idx != targeted_player_idx and void_info[targeted_player_idx][card.suit] == False:
                                for switched_card in copy_cards[targeted_player_idx]:
                                    if switched_card not in fixed_cards.get(targeted_player_idx, []) and void_info[player_idx][switched_card.suit] == False and card not in removed_cards:
                                        #print("---> player-{}'s {} card vs. player-{}'s {} card".format(player_idx, card, targeted_player_idx, switched_card))
                                        copy_cards[player_idx].append(switched_card)

                                        copy_cards[targeted_player_idx].remove(switched_card)
                                        copy_cards[targeted_player_idx].append(card)

                                        if card in remaining_cards: remaining_cards.remove(card)
                                        is_switched = True

                                        break

                            if is_switched:
                                break

                        if is_switched == False:
                            copy_cards[player_idx].append(card)
                            remaining_cards.remove(card)

            yield copy_cards
