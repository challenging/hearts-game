import random
import itertools
import math
import collections
import copy

from card import Suit, Rank

_suits = [Suit.spades, Suit.hearts, Suit.diamonds, Suit.clubs]
_nums = [Rank.two, Rank.three, Rank.four, Rank.five, Rank.six, Rank.seven,
         Rank.eight, Rank.nine, Rank.jack, Rank.queen, Rank.king, Rank.ace]

def get_all_cards():
    all_cards = list()
    for i in itertools.product(_nums,_suits):
        all_cards.append('%c%c'%(i[0],i[1]))
    return all_cards


def get_void_cards(cards_list,remain_round,remain_cards_num):
    cards = dict()
    ac = 0
    for k in remain_cards_num:
        cards[k] = cards_list[ac:ac+remain_cards_num[k]]
        ac+=remain_cards_num[k]

    void_cards = dict()
    void_cards[0] ={Suit.spades: True, Suit.hearts: True, Suit.diamonds: True, Suit.clubs: True}
    void_cards[1] ={Suit.spades: True, Suit.hearts: True, Suit.diamonds: True, Suit.clubs: True}
    void_cards[2] ={Suit.spades: True, Suit.hearts: True, Suit.diamonds: True, Suit.clubs: True}
    for k in cards:
       for c in cards[k]:
           void_cards[k][c[1]] = False
    return void_cards


def default_value():
    return 0


def get_card_combinations(total_num,separate_num,player_num_list):
    ret = collections.defaultdict(default_value)
    if separate_num == 1:
        ret[player_num_list[0]] = total_num
        yield ret
    elif separate_num == 2:
        for i in range(total_num+1):
            ret[player_num_list[0]] = i
            ret[player_num_list[1]] = total_num-i
            yield ret
    elif separate_num == 3:
        for i in range(total_num+1):
            for j in range(total_num+1-i):
                ret[player_num_list[0]] = i
                ret[player_num_list[1]] = j
                ret[player_num_list[2]] = total_num-i-j
                yield ret

def get_possible_combs(suit_dict,suit_dispatch,remain_cards_num):
       possible_suits = list()
       for suit in _suits:
           if (suit in suit_dispatch) and len(suit_dispatch[suit]) > 0:
               possible_suits.append(suit)
       all_posibility = list()
       for i in range(len(possible_suits)):
          all_posibility.append(list())
          for c in get_card_combinations(suit_dict[possible_suits[i]],len(suit_dispatch[possible_suits[i]]),suit_dispatch[possible_suits[i]]):
              all_posibility[i].append(copy.deepcopy(c))
          random.shuffle(all_posibility[i])

       for c in itertools.product(*all_posibility):
          player = collections.defaultdict(default_value)
          one_result = dict()
          for i in range(len(possible_suits)):
              for k in c[i]:
                  player[k] += c[i][k]
              one_result[possible_suits[i]]=dict(c[i])
          if player[0] == remain_cards_num[0] and player[1] == remain_cards_num[1]:
              yield one_result


def random_cards(remain_card_list,void_card_list,remain_cards_num):
    suit_dict = dict()
    suit_cards = dict()

    for card in remain_card_list:
        suit = card.suit
        if suit not in suit_dict:
            suit_dict[suit] = 1
            suit_cards[suit]=list()
        else:
            suit_dict[suit] += 1
        suit_cards[suit].append(card)

    suit_dispatch = dict()
    for i in range(3):
        for k in void_card_list[i]:
            if void_card_list[i][k] == False:
                if k not in suit_dispatch:
                    suit_dispatch[k] = list()
                suit_dispatch[k].append(i)

    return get_possible_combs(suit_dict,suit_dispatch,remain_cards_num)


def patch_remain_cards(remain_cards):
    cards_num = len(remain_cards)//3
    play_card_num = random.randint(0,3)
    remain_cards_num = {0: cards_num,1:cards_num,2:cards_num}
    play_card_num_tmp = play_card_num
    while play_card_num_tmp >0:
        remain_cards_num[3-play_card_num_tmp]-=1
        play_card_num_tmp-=1

    return remain_cards[play_card_num:],remain_cards_num


def init():
    all_cards = get_all_cards()
    random.shuffle(all_cards)
    round = random.randrange(5,13)

    remain_round = 13 - round
    remain_cards = all_cards[0:(remain_round)*4]
    my_cards = remain_cards[0:remain_round]
    other_cards_pre = remain_cards[remain_round:]
    other_cards,remain_cards_num = patch_remain_cards(other_cards_pre)
    void_cards = get_void_cards(other_cards,remain_round,remain_cards_num)
    #print "void_cards:",void_cards

    # get all possibility for specific condition
    # - other_cards
    #   ['9C', '5S', '3S', 'QC', '6S', '4C', '8H', '2C', '9S', '5H', 'JD', 'QS', '2S', 'AC', '3D']
    # - void_cards
    #   {0: {'C': False, 'H': True, 'S': False, 'D': True}, 1: {'C': False, 'H': False, 'S': False, 'D': True}, 2: {'C': False, 'H': True, 'S': False, 'D': False}}
    # - remain_cards_num:
    #   {0: 5, 1: 5, 2: 5}
    # - yield output:
    #   {'C': {0: 5, 1: 0, 2: 0}, 'H': {0: 0, 1: 2, 2: 0}, 'S': {0: 0, 1: 3, 2: 3}, 'D': {0: 0, 1: 0, 2: 2}}
    #   {'C': {0: 4, 1: 1, 2: 0}, 'H': {0: 0, 1: 2, 2: 0}, 'S': {0: 1, 1: 2, 2: 3}, 'D': {0: 0, 1: 0, 2: 2}}
    #   {'C': {0: 4, 1: 0, 2: 1}, 'H': {0: 0, 1: 2, 2: 0}, 'S': {0: 1, 1: 3, 2: 2}, 'D': {0: 0, 1: 0, 2: 2}}
    #   ...
    for i in random_cards(other_cards,void_cards,remain_cards_num):
        print(i)

if __name__ == '__main__':
    init()

