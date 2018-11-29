from card import Deck
from rules import get_rating

from nn_utils import v2card
from mcts import TreeNode, MCTS, say

SCORE_SCALAR = 52

class IntelligentTreeNode(TreeNode):
    def __init__(self, parent, prior_p, player_idx):
        super(IntelligentTreeNode, self).__init__(parent, prior_p, player_idx)


class IntelligentMCTS(MCTS):
    def __init__(self, policy_value_fn, self_player_idx, c_puct=5, min_times=0):
        super(IntelligentMCTS, self).__init__(policy_value_fn, self_player_idx, c_puct, min_times)


    def _playout(self, trick_nr, state, selection_func, c_puct):
        super(IntelligentMCTS, self)._playout(trick_nr, state, selection_func, c_puct)


    def _post_playout(self, node, trick_nr, state, selection_func, prob_cards):
        if not state.is_finished:
            probs, scores = self._policy(trick_nr, state)
            #print("probs", probs, scores)

            node.expand(state.start_pos, probs)
        else:
            scores, _ = state.score()

        node.update_recursive(get_rating(scores))


    def reinit_tree_node(self):
        self.start_node = IntelligentTreeNode(None, 1.0, None)


    def __str__(self):
        return "IntelligentMCTS"
