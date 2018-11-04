import sys

import copy
import time
import random

from random import shuffle, choice, randint

from card import Suit, SPADES_Q


def get_fixed_cards(cards, must_have):
    fixed_cards = {}

    for player_idx, cardss in must_have.items():
        fixed_cards.setdefault(player_idx, [])
        for card in cardss:
            if card in cards:
                fixed_cards[player_idx].append(card)
                cards.remove(card)

    return fixed_cards


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

            card_idx = 0
            for idx, card_1 in enumerate(copy_cards[selected_player]):
                if card_1.suit.value == random_void_suit:
                    for iidx in range(card_idx, len(copy_cards[another_selected_player])):
                        card_2 = copy_cards[another_selected_player][iidx]
                        if card_2.suit.value != random_void_suit:
                            copy_cards[selected_player][idx], copy_cards[another_selected_player][iidx] = \
                                copy_cards[another_selected_player][iidx], copy_cards[selected_player][idx]

                            break

            for selected_idx in [idx for idx in range(4) if idx != position]:
                all_hearts = all([card.suit == Suit.hearts for card in copy_cards[selected_idx]])

                if all_hearts:
                    another_selected_idx = choice([player_idx for player_idx in range(4) if player_idx != position and player_idx != selected_idx])

                    copy_cards[selected_idx][0], copy_cards[another_selected_idx][0] = \
                        copy_cards[another_selected_idx][0], copy_cards[selected_idx][0]
                """
                else:
                    num_of_suits = {}
                    for card in copy_cards[selected_idx]:
                        num_of_suits.setdefault(card.suit, [])
                        num_of_suits[card.suit].append(card)

                    if len(num_of_suits.get(Suit.spades, [])) == 1 and num_of_suits[Suit.spades][0] == SPADES_Q and len(num_of_suits.get(Suit.hearts, [])) == 12:
                        another_selected_idx = choice([player_idx for player_idx in range(4) if player_idx != position and player_idx != selected_idx])

                        for switched_card_idx in range(len(copy_cards[another_selected_idx])):
                            if copy_cards[another_selected_idx][switched_card_idx].suit != Suit.hearts:
                                copy_cards[selected_idx][0], copy_cards[another_selected_idx][switched_card_idx] = \
                                    copy_cards[another_selected_idx][switched_card_idx], copy_cards[selected_idx][0]

                                break
                """

        yield copy_cards


def redistribute_cards(seed, position, hand_cards, num_hand_cards, trick, cards, must_have={}, void_info={}, not_seen=True):
    random.seed(seed)

    fixed_cards = get_fixed_cards(cards, must_have)
    #print("numbers", num_hand_cards)

    numbers = copy.deepcopy(num_hand_cards)
    for player_idx in numbers:
        numbers[player_idx] -= len(fixed_cards.get(player_idx, []))
    del numbers[position]

    random_void = len(cards) > 19

    if sum([1 if is_void else 0 for info in void_info.values() for is_void in info.values()]) == 0 or len(cards) < 9:
        for hand_cards in simple_redistribute_cards(position, hand_cards, cards, fixed_cards, numbers, random_void):
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

                for card in remaining_cards[:]:
                    if void_info[player_idx][card.suit] == False:
                        copy_cards[player_idx].append(card)
                        remaining_cards.remove(card)

                        if len(copy_cards[player_idx]) == numbers[player_idx]+len(fixed_cards.get(player_idx, [])):
                            break

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
                                    if switched_card not in fixed_cards.get(targeted_player_idx, []) and void_info[player_idx][switched_card.suit] == False:# and card not in removed_cards:
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


if __name__ == "__main__":
    from card import Rank, Suit, Card
    from card import transform


    hand_cards = [[], [], [], [transform(card[0], card[1]) for card in "9D".split(",")]]
    trick = [[2, 32], [1, 32], [0, 64]]
    cards = [transform(card[0], card[1]) for card in "QS,2H,5H".split(",")]
    must_have = {0: [transform(card[0], card[1]) for card in "KH,7C,2C".split(",")]}
    void_info = {0: {Suit.spades: True, Suit.hearts: False, Suit.diamonds: False, Suit.clubs: False}, \
                 1: {Suit.spades: True, Suit.hearts: False, Suit.diamonds: True, Suit.clubs: True}, \
                 2: {Suit.spades: False, Suit.hearts: False, Suit.diamonds: False, Suit.clubs: False}}

    for simulation_cards in redistribute_cards(1, 3, hand_cards, trick, cards, must_have=must_have, void_info=void_info):
        print(simulation_cards)

        break
