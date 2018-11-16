import sys
import math

import redis

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


    def get_value(self, c_puct):
        u = (c_puct * self._P * math.log(self._parent._n_visits)/(1 + self._n_visits))**0.5
        q = self._Q[self._player_idx]/(1e-16+self._n_visits)
        value = q + u

        #value = self._Q[self._player_idx]/(1e-16+self._n_visits) + self._u

        return value


    def is_leaf(self):
        return self._children == {}


    def is_root(self):
        return self._parent is None



conn = redis.Redis(host='localhost', port=6379, db="12")

class RedisTreeNode(TreeNode):
    def __init__(self, parent, prior_p, player_idx, level_idx=0, card="root"):
        global conn

        self._parent = parent
        self._player_idx = player_idx

        self.level_idx = level_idx

        self.key = "{}_{}".format(self.level_idx, card)

        if not conn.exists(self.key):
            conn.hmset(self.key, {"parent": self._parent, 
                                  "p": prior_p, 
                                  "n_visits": 0, 
                                  "q_1": 0.0,
                                  "q_2": 0.0,
                                  "q_3": 0.0,
                                  "q_4": 0.0})

            if self._parent is not None:
                conn.lpush("{}:children".format(self._parent), self.key)


    def get_children(self):
        global conn

        return conn.lrange("{}:children".format(self.key), 0, -1)


    def expand(self, player_idx, action_priors):
        global conn

        children = set(self.get_children())
        for action, prob in action_priors:
            card = bitmask_to_str(action[0], action[1])
            key = "{}_{}".format(self.level_idx+1, card)

            if not conn.exists(key):
                RedisTreeNode(self.key, prob, player_idx, level_idx=self.level_idx+1, card=card)


    def get_info(self, fields, key=None):
        global conn

        return conn.hmget(key if key else self.key, fields)


    def select(self, c_puct):
        children = self.get_children()

        parent_n_visits = float(self.get_info(["n_visits"])[0])

        results = {}
        for node in children:
            card = str(node).strip("'").split("_")[1]
            key = (SUIT_TO_INDEX[card[1]], NUM_TO_INDEX[card[0]])

            results[key] = self.get_value(node, c_puct, parent_n_visits)

        for card, value in sorted(results.items(), key=lambda x: -x[1]):
            yield card, value


    def reset_player_idx(self, player_idx):
        self._player_idx = player_idx


    def update(self, leaf_value):
        global conn

        info = self.get_info(["q_{}".format(self._player_idx), "n_visits"])
        print("before", info)
        conn.hmset(self.key, {"q_{}".format(self._player_idx): float(info[0])+leaf_value, "n_visits": int(info[1])+1})

        info = self.get_info(["q_{}".format(self._player_idx), "n_visits"])
        print(" after", info)


    def update_recursive(self, scores):
        if self._parent:
            #self._parent.update_recursive(scores)
            RedisTreeNode(self.key, prob, player_idx, level_idx=self.level_idx+1, card=card).update_recursive(scores)

        self.update(scores[self._player_idx])


    def get_value(self, node, c_puct, parent_n_visits=None):
        global conn

        if parent_n_visits is None:
            parent_n_visits = float(self.get_info(["n_visits"])[0])

        info = self.get_info(["n_visits", "q_{}".format(self._player_idx), "p"], node)
        n_visits, q_value, p = int(info[0]), float(info[1]), float(info[2])

        value = q_value/(1e-16+n_visits) + (c_puct * p * math.log(parent_n_visits+1)/(1 + n_visits))**0.5

        return value


    


if __name__ == "__main__":
    import time

    stime = time.time()

    tree = RedisTreeNode(None, 1.0, None)
    tree.expand(1, [((0, 1), 1.0)])

    tree = RedisTreeNode("1_2C", 1.0, 2, 2, "4C")
    tree = RedisTreeNode("1_2C", 1.0, 2, 2, "6C")

    tree.update_recursive([1, 2, 3, 4])

    #tree = RedisTreeNode("0_root", 1.0, 1, 1, "2C")

    #for card, node in tree.select(1024):
    #    print(card, node)

    print(time.time()-stime)
