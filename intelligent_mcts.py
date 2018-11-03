from card import Deck
from rules import get_rating

from mcts import TreeNode, MCTS


class IntelligentTreeNode(TreeNode):
    def __init__(self, parent, prior_p, self_player_idx, player_idx):
        super(IntelligentTreeNode, self).__init__(parent, prior_p, self_player_idx, player_idx)



class IntelligentMCTS(MCTS):
    def __init__(self, policy_value_fn, self_player_idx, c_puct=5):
        super(IntelligentMCTS, self).__init__(policy_value_fn, self_player_idx, c_puct)


    def _post_playout(self, node, trick_nr, state, selection_func):
        scores = [0, 0, 0, 0]
        if not state.is_finished:
            probs, scores = self._policy(trick_nr, state)
            action_probs = zip([(card.suit.value, 1 << (card.rank.value-2)) for card in Deck().cards], probs)

            node.expand(state.start_pos, action_probs)
        else:
            scores = get_rating(state.score()[0])

        node.update_recursive(scores)


    def __str__(self):
        return "IntelligentMCTS"
