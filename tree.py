import sys
import math

from card import NUM_TO_INDEX, SUIT_TO_INDEX
from card import FULL_CARDS, NUM_TO_INDEX, SUIT_TO_INDEX

class TreeNode(object):
    def __init__(self, parent, prior_p, player_idx):
        self._parent = parent

        self._player_idx = player_idx

        self._children = {}
        self._n_visits = 0
        self._Q = [0, 0, 0, 0]
        self._u = 0
        self._P = prior_p


    def expand(self, player_idx, action_priors):
        #print("expansion")
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, player_idx)
            else:
                self._children[action]._player_idx = player_idx
                self._children[action]._P = prob


    def select(self, c_puct):
        for card, node in sorted(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)*-1):
            yield card, node


    def update(self, leaf_value):
        self._n_visits += 1

        self._Q[self._player_idx] += leaf_value


    def update_recursive(self, scores):
        if self._parent:
            self._parent.update_recursive(scores)

        if self._player_idx is None:
            self._n_visits += 1
        else:
            self.update(scores[self._player_idx])


    def update_recursive_percentage(self, probs):
        for action, node in self._children.items():
            node.update_recursive_percentage(probs)

            suit, rank = action
            for (s, r), prob in probs:
                if suit == s and rank == r:
                    node._P = prob

                    break


    def get_value(self, c_puct):
        self._u = (c_puct * self._P * math.log(self._parent._n_visits)/(1 + self._n_visits))**0.5
        value = self._Q[self._player_idx]/(1e-16+self._n_visits) + self._u

        return value


    def is_leaf(self):
        return self._children == {}


    def is_root(self):
        return self._parent is None



class LevelNode(object):
    def __init__(self, parent, player_idx, level, is_init=True):
        self._parent = parent
        self._child = None

        self._probs = {}

        self._player_idx = player_idx
        self.level = level

        if is_init:
            self.setup()


    def setup(self):
        node = self

        num_round = 52
        for step_idx in range(num_round):
            results = []
            if step_idx == 0:
                for suit, ranks in FULL_CARDS.items():
                    bitmask = NUM_TO_INDEX["2"]
                    while bitmask <= NUM_TO_INDEX["A"]:
                        if suit == SUIT_TO_INDEX["C"] and bitmask == NUM_TO_INDEX["2"]:
                            results.append([(suit, bitmask), 1.0])
                        else:
                            results.append([(suit, bitmask), 0.0])

                        bitmask <<= 1
            elif step_idx < 4:
                prob = 1/(num_round-step_idx-14)

                for suit, ranks in FULL_CARDS.items():
                    bitmask = NUM_TO_INDEX["2"]
                    while bitmask <= NUM_TO_INDEX["A"]:
                        if suit == SUIT_TO_INDEX["C"] and bitmask == NUM_TO_INDEX["2"]:
                            results.append([(suit, bitmask), 0.0])
                        elif suit == SUIT_TO_INDEX["S"] and bitmask == NUM_TO_INDEX["Q"]:
                            results.append([(suit, bitmask), 0.0])
                        elif suit == SUIT_TO_INDEX["H"]:
                            results.append([(suit, bitmask), 0.0])
                        else:
                            results.append([(suit, bitmask), prob])

                        bitmask <<= 1
            else:
                prob = 1/(num_round-step_idx)

                for suit, ranks in FULL_CARDS.items():
                    bitmask = NUM_TO_INDEX["2"]
                    while bitmask <= NUM_TO_INDEX["A"]:
                        if suit == SUIT_TO_INDEX["C"] and bitmask == NUM_TO_INDEX["2"]:
                            results.append([(suit, bitmask), 0.0])
                        else:
                            results.append([(suit, bitmask), prob])

                        bitmask <<= 1

            node.expand(self._player_idx, results)
            #print(self, node.level, node, node._child)

            node = node._child


    def expand(self, player_idx, action_priors):
        for action, prob in action_priors:
            self._probs[action] = [0, prob, 0]

        self._child = LevelNode(self, player_idx, self.level+1, is_init=False)


    def get_value(self, c_puct, parent_visits, info):
        q_value, p_value, n_visits = info[0], info[1], info[2]

        u = (c_puct * p_value * (parent_visits)**0.5 / (1 + n_visits))

        return q_value/(1e-16+n_visits) + u


    def select(self, parent_card, c_puct, ascending=False):
        order = 1 if ascending else -1

        parent_visits = self._probs[parent_card][2]
        for card, _ in sorted(self._probs.items(), key=lambda act_node: self.get_value(c_puct, parent_visits, act_node[1])*order):
            yield card, self._child


    def update(self, path, leaf_value):
        self._probs[path][0] += leaf_value
        self._probs[path][2] += 1


    def update_recursive(self, paths, scores):
        path = paths.pop()

        if self._parent:
            self._parent.update_recursive(paths, scores)

        self.update(path, scores[self._player_idx])


    def update_recursive_percentage(self, probs):
        for card, prob in probs:
            self._probs[card][1] = prob

        if self._child:
            update_recursive_percentage(self._child, probs)


    def is_leaf(self):
        return self._child is None


    def is_root(self):
        return self._parent is None
