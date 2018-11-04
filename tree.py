

class TreeNode(object):
    def __init__(self, parent, prior_p, self_player_idx, player_idx):
        self._parent = parent

        self._self_player_idx = self_player_idx
        self._player_idx = player_idx

        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p


    def expand(self, player_idx, action_priors):
        #print("expansion")
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, self._self_player_idx, player_idx)
            else:
                self._children[action]._player_idx = player_idx
                self._children[action]._P = prob


    def select(self, c_puct):
        for card, node in sorted(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct)*-1):
            yield card, node


    def update(self, leaf_value):
        self._n_visits += 1

        self._Q += leaf_value


    def update_recursive(self, scores):
        if self._parent:
            self._parent.update_recursive(scores)

        if self._player_idx is None:
            self._n_visits += 1
        else:
            v = scores[self._player_idx]
            #print(self._self_player_idx, self._player_idx, scores, v)

            self.update(v)


    def update_recursive_percentage(self, probs):
        for action, node in self._children.items():
            node.update_recursive_percentage(probs)

            suit, rank = action
            for (s, r), prob in probs:
                if suit == s and rank == r:
                    node._P = prob

                    break


    def get_value(self, c_puct):
        self._u = (c_puct * self._P * (self._parent._n_visits)**0.5 / (1 + self._n_visits))

        return self._Q/(1e-16+self._n_visits) + self._u


    def is_leaf(self):
        return self._children == {}


    def is_root(self):
        return self._parent is None
