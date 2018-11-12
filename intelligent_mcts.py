from card import Deck
from rules import get_rating

from mcts import TreeNode, MCTS


class IntelligentTreeNode(TreeNode):
    def __init__(self, parent, prior_p, player_idx):
        super(IntelligentTreeNode, self).__init__(parent, prior_p, player_idx)


class IntelligentMCTS(MCTS):
    def __init__(self, policy_value_fn, self_player_idx, c_puct=5, min_times=256):
        super(IntelligentMCTS, self).__init__(policy_value_fn, self_player_idx, c_puct, min_times)


    def _playout(self, trick_nr, state, selection_func, c_puct):
        super(IntelligentMCTS, self)._playout(trick_nr, state, selection_func, c_puct)


    def _post_playout(self, node, trick_nr, state, selection_func, prob_cards):
        scores = [0, 0, 0, 0]
        if not state.is_finished:
            probs, scores = self._policy(trick_nr, state)
            action_probs = zip([(card.suit.value, 1 << (card.rank.value-2)) for card in Deck().cards], probs)

            node.expand(state.start_pos, action_probs)
        else:
            scores, is_shootthemoon = state.score()

        node.update_recursive(get_rating(scores))


    def update_with_move(self, last_move):
        if last_move in self.start_node._children:
            self.start_node = self.start_node._children[last_move]
        else:
            self.start_node = TreeNode(None, 1.0, None)


    def __str__(self):
        return "IntelligentMCTS"
