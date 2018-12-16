import sys
import math

from card import NUM_TO_INDEX, SUIT_TO_INDEX
from card import FULL_CARDS, NUM_TO_INDEX, SUIT_TO_INDEX

from card import bitmask_to_str, str_to_bitmask


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
        #print("expand", player_idx, action_priors)
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, player_idx)
            else:
                self._children[action]._player_idx = player_idx
                self._children[action]._P = (self._children[action]._P + prob)/2


    def select(self, c_puct):
        for card, node in sorted(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)*-1):
            yield card, node


    def update(self, leaf_value):
        self._n_visits += 1

        self._Q[self._player_idx] += leaf_value
        #self._Q += leaf_value


    def update_recursive(self, scores):
        if self._parent:
            self._parent.update_recursive(scores)

        if self._player_idx is None:
            self._n_visits += 1
        else:
            self.update(scores[self._player_idx])


    def get_value(self, c_puct):
        u = c_puct * self._P * (math.log(self._parent._n_visits)/(1 + self._n_visits))**0.5
        #u = (c_puct * self._P * (self._parent._n_visits)**0.5 / (1 + self._n_visits))
        q = self._Q[self._player_idx]/(1e-16+self._n_visits)
        #q = self._Q / (1e-16+self._n_visits)

        value = q + u
        #print("q={:2.8f}, u={:.8f}, c_puct={:.1f}, p={:.4f}, n_parent_visits={}, n_visits={}".format(\
        #    q, u, c_puct, self._P, self._parent._n_visits, self._n_visits))
        #value = self._Q[self._player_idx]/(1e-16+self._n_visits) + self._u

        return value


    def is_leaf(self):
        return self._children == {}


    def is_root(self):
        return self._parent is None
