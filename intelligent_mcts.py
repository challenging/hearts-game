from card import Deck
from rules import get_rating

from nn_utils import v2card
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
        scores = None
        if state.is_finished:
            scores, is_shootthemoon = state.score()
        else:
            bcards, probs, scores = self._policy(trick_nr, state)
            valid_cards = [v2card(v) for v in bcards if v > 0]
            action_probs = zip([(card.suit.value, 1<<(card.rank.value-2)) for card in valid_cards], probs)

            node.expand(state.start_pos, action_probs)

        node.update_recursive(get_rating(scores))


    def __str__(self):
        return "IntelligentMCTS"
